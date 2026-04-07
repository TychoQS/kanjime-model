import torch
import os
from tqdm import tqdm
from .evaluation import calculate_iou

def run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, config, early_stop, scheduler, checkpoint_path, history_path):
    start_epoch = 0
    best_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['best_iou']
        history = checkpoint['history']

    for epoch in range(start_epoch, config.EPOCHS):
        # Training
        model.train()
        t_loss, t_iou = 0, 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]", leave=False)
        
        for imgs, masks in pbar_train:
            imgs = imgs.to(device).float()
            masks = masks.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            t_iou += calculate_iou(outputs, masks)
            
        avg_t_loss = t_loss / len(train_loader)
        avg_t_iou = t_iou / len(train_loader)

        # Validation
        model.eval()
        v_loss, v_iou = 0, 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        
        with torch.no_grad():
            for imgs, masks in pbar_val:
                imgs = imgs.to(device).float()
                masks = masks.to(device).float()
                
                outputs = model(imgs)
                v_loss += criterion(outputs, masks).item()
                v_iou += calculate_iou(outputs, masks)
        
        avg_v_loss = v_loss / len(val_loader)
        avg_v_iou = v_iou / len(val_loader)

        # Update History & Scheduler
        scheduler.step(avg_v_loss)
        history["train_loss"].append(avg_t_loss)
        history["val_loss"].append(avg_v_loss)
        history["train_iou"].append(avg_t_iou)
        history["val_iou"].append(avg_v_iou)

        print(f"Epoch {epoch+1} -> Train IoU: {avg_t_iou:.4f} | Val IoU: {avg_v_iou:.4f} | Val Loss: {avg_v_loss:.4f}")

        # Save Best Model Weights
        if avg_v_iou > best_iou:
            best_iou = avg_v_iou
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "best_model.pth"))

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
            'history': history
        }, checkpoint_path)

        early_stop(avg_v_loss)
        if early_stop.early_stop:
            print("Early Stopping triggered.")
            break
            
    return history

def run_test_evaluation(model, loader, criterion, device, config):
    best_model_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    model.eval()
    test_loss, test_iou = 0, 0
    
    print(f"Starting test evaluation on {device}...")
    
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Test Set Evaluation"):
            imgs = imgs.to(device).float()
            masks = masks.to(device).float()
            
            outputs = model(imgs)
            
            test_loss += criterion(outputs, masks).item()
            test_iou += calculate_iou(outputs, masks)
            
    avg_loss = test_loss / len(loader)
    avg_iou = test_iou / len(loader)
    
    return avg_loss, avg_iou