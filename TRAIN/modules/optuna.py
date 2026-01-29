import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import gc

def objective(trial, get_dataloaders_fn, build_model_fn, device, optuna_epochs):
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 96, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    t_loader, v_loader, n_clases = get_dataloaders_fn(batch_size)
    trial_model = build_model_fn(n_clases)
    
    optimizer = optim.AdamW(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    
    best_val_acc = 0.0
    
    for epoch in range(optuna_epochs):
        trial_model.train()
        for inputs, labels in t_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = trial_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        trial_model.eval()
        corrects = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in v_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = trial_model(inputs)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data).item()
                total += labels.size(0)
        
        val_acc = corrects / total
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    del trial_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_val_acc