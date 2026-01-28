import os
import torch
from tqdm import tqdm
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, output_dir, model_save_path, device, early_stopping, scheduler=None, start_epoch=0, best_acc=0.0, history=None):    
    """Training function for the model"""
    if history is None: # Create history if not provided (not resuming training)
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    print(f"Saving results to: {output_dir}")
    
    for epoch in range(start_epoch, num_epochs): 
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training Phase
        model.train() # Set model to training mode (enables Dropout/BatchNorm)
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Clear gradients from previous step
            
            # Forward pass: Compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
            loss = criterion(outputs, labels) # Calculate loss
        
            loss.backward() # Compute gradients
            
            optimizer.step() # Updates network weights based on gradients

            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Training stats
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double().item() / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation
        model.eval() # Set model to evaluation mode 
        val_loss_val = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss_val += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        # Validation stats
        val_loss_epoch = val_loss_val / len(val_loader.dataset)
        val_acc = val_corrects.double().item() / len(val_loader.dataset)
        history['val_loss'].append(val_loss_epoch)
        history['val_acc'].append(val_acc)

        print(f'Val Loss: {val_loss_epoch:.4f} Acc: {val_acc:.4f}')
        if scheduler: 
            scheduler.step(val_acc) # Update scheduler
            print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        if early_stopping(val_acc): break # Stop if no improvement

        checkpoint = { # Save point
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_acc': best_acc
        }
        torch.save(checkpoint, os.path.join(output_dir, 'last_checkpoint.pth'))

        if val_acc > best_acc: # Save best model
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path) 
            print(f"Model saved: {model_save_path}")

    print('\nTraining finished.')
    return model, history
