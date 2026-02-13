import os
import torch
import time
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import logging
from modules.train_utils import CurriculumManager

logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, criterion_kanji, criterion_radicals, criterion_strokes, optimizer, num_epochs, output_dir, model_save_path, device, early_stopping, scheduler=None, start_epoch=0, best_acc=0.0, history=None, save_data=True, verbose=True, lambda_rad=1.0, lambda_str=1.0):    
    """Training function for the model"""

    curriculum_manager = CurriculumManager(threshold=0.75)
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    if history is None: # Create history if not provided (not resuming training)
        history = {
            'train_loss': [], 'train_acc': [], 'train_radical_acc': [], 'train_stroke_acc': [],
            'val_loss': [], 'val_acc': [], 'val_radical_acc': [], 'val_stroke_acc': []
        }
    if save_data: logger.info(f"Saving results to: {output_dir}")
    
    for epoch in range(start_epoch, num_epochs): 
        logger.info(f'\nEpoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)

        # Training Phase
        model.train() # Set model to training mode (enables Dropout/BatchNorm)
        running_loss = 0.0
        running_corrects = 0
        running_corrects_radical = 0
        running_corrects_strokes = 0

        for inputs, labels in tqdm(train_loader, desc="Training", disable=not verbose):
            inputs = inputs.to(device)
            labels_kanji = labels['kanji'].to(device)
            labels_radical = labels['radical'].to(device)
            labels_strokes = labels['strokes'].to(device)

            optimizer.zero_grad() # Clear gradients from previous step
            
            # Forward pass: Compute predicted outputs by passing inputs to the model
            outputs_kanji, outputs_radical, outputs_strokes = model(inputs)
            _, preds = torch.max(outputs_kanji, 1)
            _, preds_radical = torch.max(outputs_radical, 1)
            _, preds_strokes = torch.max(outputs_strokes, 1)
        
            # Calculate losses
            loss_kanji = criterion_kanji(outputs_kanji, labels_kanji)
            loss_radical = criterion_radicals(outputs_radical, labels_radical)
            loss_strokes = criterion_strokes(outputs_strokes, labels_strokes)
            loss = (loss_kanji + lambda_rad * loss_radical + lambda_str * loss_strokes) if curriculum_manager.is_kanji_active() else (lambda_rad * loss_radical + lambda_str * loss_strokes)        
            loss.backward() # Compute gradients
            optimizer.step() # Updates network weights based on gradients

            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels_kanji.data)
            running_corrects_radical += torch.sum(preds_radical == labels_radical.data)
            running_corrects_strokes += torch.sum(preds_strokes == labels_strokes.data)

        # Training stats
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double().item() / len(train_loader.dataset)
        epoch_radical_acc = running_corrects_radical.double().item() / len(train_loader.dataset)
        epoch_stroke_acc = running_corrects_strokes.double().item() / len(train_loader.dataset)
        
        history['train_radical_acc'].append(epoch_radical_acc)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_stroke_acc'].append(epoch_stroke_acc)

        logger.info(f'Train Loss: {epoch_loss:.4f} | Kanji Acc: {epoch_acc:.4f} | Radical Acc: {epoch_radical_acc:.4f} | Stroke Acc: {epoch_stroke_acc:.4f}')

        # Validation
        model.eval() # Set model to evaluation mode 
        val_loss_val = 0.0
        val_corrects = 0
        val_corrects_radical = 0
        val_corrects_strokes = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                l_kanji = labels['kanji'].to(device)
                l_radical = labels['radical'].to(device)
                l_strokes = labels['strokes'].to(device)

                out_k, out_r, out_s = model(inputs)
                _, preds = torch.max(out_k, 1)
                _, preds_radical = torch.max(out_r, 1)
                _, preds_strokes = torch.max(out_s, 1)
                loss = (criterion_kanji(out_k, l_kanji) + lambda_rad * criterion_radicals(out_r, l_radical) + lambda_str * criterion_strokes(out_s, l_strokes)) if curriculum_manager.is_kanji_active() else (lambda_rad * criterion_radicals(out_r, l_radical) + lambda_str * criterion_strokes(out_s, l_strokes))
                val_loss_val += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == l_kanji.data)
                val_corrects_radical += torch.sum(preds_radical == l_radical.data)
                val_corrects_strokes += torch.sum(preds_strokes == l_strokes.data)

        # Validation stats
        val_loss_epoch = val_loss_val / len(val_loader.dataset)
        val_acc = val_corrects.double().item() / len(val_loader.dataset)
        val_radical_acc = val_corrects_radical.double().item() / len(val_loader.dataset)
        val_stroke_acc = val_corrects_strokes.double().item() / len(val_loader.dataset)

        history['val_radical_acc'].append(val_radical_acc)
        history['val_loss'].append(val_loss_epoch)
        history['val_acc'].append(val_acc)
        history['val_stroke_acc'].append(val_stroke_acc)

        curriculum_manager.update(val_radical_acc, val_stroke_acc)
        if not curriculum_manager.is_kanji_active():
            if (epoch - start_epoch) >= 15:
                logger.error(f"Training aborted: After {15} epochs, "
                             f"Radicals ({val_radical_acc:.2f}) or Strokes ({val_stroke_acc:.2f}) "
                             f"have not reached the threshold of {curriculum_manager.threshold}.")
                break
        logger.info(f'Val Loss: {val_loss_epoch:.4f} | Kanji Acc: {val_acc:.4f} | Radical Acc: {val_radical_acc:.4f} | Stroke Acc: {val_stroke_acc:.4f}')

        checkpoint = { # Save point
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_acc': best_acc
        }
        if save_data: torch.save(checkpoint, os.path.join(output_dir, 'last_checkpoint.pth'))

        if curriculum_manager.is_kanji_active():
            if scheduler: 
                scheduler.step(val_acc) # Update scheduler
                logger.info(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            if early_stopping(val_acc): break # Stop if no improvement

            if val_acc > best_acc: # Save best model
                best_acc = val_acc
                if save_data: torch.save(model.state_dict(), model_save_path) 
                logger.info(f"Model saved: {model_save_path}")

    logger.info('\nTraining finished.')
    if save_data: model.load_state_dict(torch.load(model_save_path))
    return model, history