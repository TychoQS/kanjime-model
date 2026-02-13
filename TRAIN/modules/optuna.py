import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import gc
from modules.train_utils import setup_training_tools    
from modules.train_model import train_model
from modules.train_utils import EarlyStopping

def objective(trial, get_dataloaders_fn, build_model_fn, device, optuna_epochs, n_clases, n_radicals, n_strokes):
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 96, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    lambda_rad = trial.suggest_float("lambda_rad", 0.1, 1)
    lambda_str = trial.suggest_float("lambda_str", 0.1, 1)
    
    t_loader, v_loader = get_dataloaders_fn(batch_size=batch_size)
    trial_model = build_model_fn(n_clases, n_radicals, n_strokes)
    
    optimizer, scheduler, criterion_kanji, criterion_radicals, criterion_strokes = setup_training_tools(trial_model, lr, weight_decay)
    
    early_stopping = EarlyStopping(patience=5, verbose=False)

    _, history = train_model(
        model=trial_model,
        train_loader=t_loader,
        val_loader=v_loader,
        criterion_kanji=criterion_kanji,
        criterion_radicals=criterion_radicals,
        criterion_strokes=criterion_strokes,
        optimizer=optimizer,
        num_epochs=optuna_epochs,
        output_dir="",           # No saving data for optuna so is irrelevant
        model_save_path="",      # No saving data for optuna so is irrelevant
        device=device,
        early_stopping=early_stopping,
        scheduler=scheduler,
        save_data=False,  
        verbose=False,
        lambda_rad=lambda_rad,
        lambda_str=lambda_str     
    )
    
    best_val_acc = max(history['val_acc'])
    
    del trial_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_val_acc