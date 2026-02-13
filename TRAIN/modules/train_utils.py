import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

class EarlyStopping:
    """Early Stopping implementation for training"""
    def __init__(self, patience=7, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.verbose = verbose
        
    def __call__(self, val_accuracy):
        score = val_accuracy
        
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f"  [Early Stopping] Baseline set: {score:.4f}")
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"  [Early Stopping] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"  [Early Stopping] TRIGGERED! Best score: {self.best_score:.4f}")
                return True
        else:
            if self.verbose:
                print(f"  [Early Stopping] Improvement! {self.best_score:.4f} → {score:.4f}")
            self.best_score = score
            self.counter = 0
        return False

def setup_training_tools(model, lr, weight_decay, factor=0.5, patience=2):
    """
    Centralize the Optimizer, Scheduler and Criterion instantiation in a single function.
    """

    trainable_params = filter(lambda p: p.requires_grad, model.parameters()) # Filter out non-trainable parameters (by default all parameters are trainable)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Max val_accuracy
        factor=factor, # Decrease LR to factor of it's value
        patience=patience, # Wait for patience epochs before reducing LR
        min_lr=1e-6 # Minimum LR value
    )
    criterion_kanji = nn.CrossEntropyLoss()
    criterion_radicals = nn.CrossEntropyLoss()
    criterion_strokes = nn.CrossEntropyLoss()
    
    return optimizer, scheduler, criterion_kanji, criterion_radicals, criterion_strokes


class CurriculumManager:
    """Manages the curriculum of the training process"""
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self._is_kanji_active = False

    def update(self, acc_rad, acc_str):
        if not self._is_kanji_active:
            if acc_rad >= self.threshold and acc_str >= self.threshold:
                print(f"Target reached! (Rad: {acc_rad:.2f}, Str: {acc_str:.2f}). Activating Kanji learning.")
                self._is_kanji_active = True
        return self._is_kanji_active
    
    def is_kanji_active(self):
        return self._is_kanji_active