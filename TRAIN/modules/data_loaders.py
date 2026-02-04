import torch
from torch.utils.data import DataLoader, Subset
from modules.config import RANDOM_SEED

def create_splits(total_size, train_ratio, val_ratio):
    """Creates the train, validation and test splits"""

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    print(f"Total: {total_size}")
    print(f"Split: Train={train_size}, Val={val_size}, Test={test_size}")

    # Create indexes for splitting
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indexes = torch.randperm(total_size, generator=generator).tolist()
    train_idx = indexes[:train_size]
    val_idx = indexes[train_size : train_size + val_size]
    test_idx = indexes[train_size + val_size :]
    return train_idx, val_idx, test_idx

def get_dataloaders(train_subset, val_subset, test_subset, batch_size, num_workers=2):
    """Returns the train, validation and test dataloaders with a custom batch size and number of workers"""
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

def get_dataloaders_reduced(full_dataset, train_dataset_augmented, batch_size, samples_per_class=50):
    """
    A reduced dataset for optuna hyperparameter optimization. 
    Samples_per_class is used to determine the amount of samples for each class
    """
    filtered_indexes = []
    for class_name in full_dataset.classes:
        class_samples_indexes = [
            i for i, sample in enumerate(full_dataset.samples) 
            if sample[1] == class_name
        ]
        selected_indexes = class_samples_indexes[:samples_per_class]
        filtered_indexes.extend(selected_indexes)
    
    subset_optuna = Subset(train_dataset_augmented, filtered_indexes)
    train_size_opt = int(0.8 * len(subset_optuna))
    val_size_opt = len(subset_optuna) - train_size_opt
    
    train_opt, val_opt = torch.utils.data.random_split(
        subset_optuna, [train_size_opt, val_size_opt]
    )

    return (
        DataLoader(train_opt, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(val_opt, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        len(full_dataset.classes),
        len(full_dataset.comp_to_idx)
    )