import os
import torch
import time
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

def train_model(model, train_loader, val_loader, criterion_kanji, criterion_components, optimizer, num_epochs, output_dir, model_save_path, device, early_stopping, scheduler=None, start_epoch=0, best_acc=0.0, history=None):    
    """Training function for the model"""
    if history is None: # Create history if not provided (not resuming training)
        history = {
        'train_loss': [], 'train_acc': [], 'train_comp_acc': [],
        'val_loss': [], 'val_acc': [], 'val_comp_acc': []
        }
    print(f"Saving results to: {output_dir}")
    
    for epoch in range(start_epoch, num_epochs): 
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training Phase
        model.train() # Set model to training mode (enables Dropout/BatchNorm)
        running_loss = 0.0
        running_corrects = 0
        running_corrects_components = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels_kanji = labels['kanji'].to(device)
            labels_components = labels['components'].to(device)

            optimizer.zero_grad() # Clear gradients from previous step
            
            # Forward pass: Compute predicted outputs by passing inputs to the model
            outputs_kanji, outputs_components = model(inputs)
            _, preds = torch.max(outputs_kanji, 1)
            preds_components = (torch.sigmoid(outputs_components) > 0.5).float()
        
            # Calculate losses
            loss_kanji = criterion_kanji(outputs_kanji, labels_kanji)
            loss_components = criterion_components(outputs_components, labels_components)
            loss = loss_kanji + loss_components
        
            loss.backward() # Compute gradients
            
            optimizer.step() # Updates network weights based on gradients

            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels_kanji.data)
            exact_match = (preds_components == labels_components).all(dim=1).float()
            running_corrects_components += exact_match.sum()

        # Training stats
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double().item() / len(train_loader.dataset)
        epoch_comp_acc = running_corrects_components.double().item() / len(train_loader.dataset)
        
        history['train_comp_acc'].append(epoch_comp_acc)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        print(f'Train Loss: {epoch_loss:.4f} | Kanji Acc: {epoch_acc:.4f} | Comp Acc: {epoch_comp_acc:.4f}')

        # Validation
        model.eval() # Set model to evaluation mode 
        val_loss_val = 0.0
        val_corrects = 0
        val_corrects_components = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                l_kanji = labels['kanji'].to(device)
                l_comp = labels['components'].to(device)

                out_k, out_c = model(inputs)
                _, preds = torch.max(out_k, 1)
                preds_comp = (torch.sigmoid(out_c) > 0.5).float()
                loss = criterion_kanji(out_k, l_kanji) + criterion_components(out_c, l_comp)
                val_loss_val += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == l_kanji.data)
                exact_match_val = (preds_comp == l_comp).all(dim=1).float()
                val_corrects_components += exact_match_val.sum()

        # Validation stats
        val_loss_epoch = val_loss_val / len(val_loader.dataset)
        val_acc = val_corrects.double().item() / len(val_loader.dataset)
        val_comp_acc = val_corrects_components.double().item() / len(val_loader.dataset)

        history['val_comp_acc'].append(val_comp_acc)
        history['val_loss'].append(val_loss_epoch)
        history['val_acc'].append(val_acc)

        print(f'Val Loss: {val_loss_epoch:.4f} | Kanji Acc: {val_acc:.4f} | Comp Acc: {val_comp_acc:.4f}')
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
    model.load_state_dict(torch.load(model_save_path))
    return model, history


def train_kaggle(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='cpu', model_save_path=None):
    """Artem Bortezko kaggle train method"""
    print('train() called: Model = %s, Optimizer = %s(lr=%f), Epochs = %d, Device = %s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    best_train_acc = 0
    best_val_acc = 0

    start_time_sec = time.time()

    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        train_loss = 0.0
        num_train_correct = 0
        num_train_examples = 0
        
        epoch_start_time_sec = time.time()

        for batch in train_dl:
            optimizer.zero_grad()
            x = batch[0].to(device)
            y = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.data.item() * x.size(0)
            num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc = num_train_correct / num_train_examples
        train_loss = train_loss / len(train_dl.dataset)

        model.eval()
        val_loss = 0.0
        num_val_correct = 0
        num_val_examples = 0

        for batch in val_dl:
            x = batch[0].to(device)
            y = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss += loss.data.item() * x.size(0)
            num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)
        
        epoch_end_time_sec = time.time()
        epoch_time_sec = epoch_end_time_sec - epoch_start_time_sec

        print(f"Epoch {epoch}, time: {round(epoch_time_sec)} seconds. Train loss: {round(train_loss, 3)}, train accuracy: {round(train_acc, 3)}. Val loss: {round(val_loss, 3)}, val accuracy: {round(val_acc, 3)}")
        
        if train_acc > best_train_acc and val_acc > best_val_acc:
          best_train_acc = train_acc
          best_val_acc = val_acc
          if model_save_path is not None:
            torch.save(model.state_dict(), model_save_path)
          best_epoch = epoch

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    end_time_sec = time.time()
    total_time_sec = end_time_sec - start_time_sec
    print("\n")
    print(f"Training Time: {round(total_time_sec)}")
    print(f"The best epoch: {best_epoch}, Train Accuracy: {round(best_train_acc, 3)}, Validation Accuracy: {round(best_val_acc, 3)}")
    model.load_state_dict(torch.load(model_save_path))
    return model, history