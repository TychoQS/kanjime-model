import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules.train_utils import setup_training_tools

def objective(trial, get_dataloaders_fn, build_model_fn, device, optuna_epochs):
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 96, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    t_loader, v_loader, n_clases, n_components = get_dataloaders_fn(batch_size=batch_size)
    trial_model = build_model_fn(n_clases, n_components)
    
    optimizer, scheduler, criterion_kanji, criterion_components = setup_training_tools(trial_model, lr, weight_decay)
    
    best_val_acc = 0.0
    
    for epoch in range(optuna_epochs):
        trial_model.train()
        for inputs, labels in t_loader:
            inputs = inputs.to(device)
            labels_kanji = labels['kanji'].to(device)
            labels_components = labels['components'].to(device)
            optimizer.zero_grad()
            outputs_kanji, outputs_components = trial_model(inputs)
            loss_kanji = criterion_kanji(outputs_kanji, labels_kanji)
            loss_components = criterion_components(outputs_components, labels_components)
            loss = loss_kanji + loss_components
            loss.backward()
            optimizer.step()
        
        trial_model.eval()
        corrects = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in v_loader:
                inputs = inputs.to(device)
                labels_kanji = labels['kanji'].to(device)
                labels_components = labels['components'].to(device)
                
                outputs_kanji, outputs_components = trial_model(inputs) 
                _, preds = torch.max(outputs_kanji, 1)                  
                corrects += torch.sum(preds == labels_kanji.data).item()
                total += labels_kanji.size(0)
        
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