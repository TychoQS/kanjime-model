import torch
import json
import os
import segmentation_models_pytorch as smp
from modules import config, dataset, architecture, transforms, utils, train_model

def main():
    # Setup environment
    utils.seed_everything(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    checkpoint_path = os.path.join(config.OUTPUT_DIR, "checkpoint.pth")
    history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
    plot_path = os.path.join(config.OUTPUT_DIR, "training_curves.png")

    # Dataset
    base_trans = transforms.get_base_transforms()
    train_ds = dataset.SyntheticKanjiDataset(config.DATASET_ROOT, split="train", transform=base_trans)
    val_ds = dataset.SyntheticKanjiDataset(config.DATASET_ROOT, split="val", transform=base_trans)
    test_ds = dataset.SyntheticKanjiDataset(config.DATASET_ROOT, split="test", transform=base_trans)
    
    # Loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize Model and Tools
    model = architecture.get_unet_model().to(device)
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config.SCHEDULER_PATIENCE)
    early_stop = utils.EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    # Training Loop
    history = train_model.run_training_loop(
        model, train_loader, val_loader, optimizer, criterion, 
        device, config, early_stop, scheduler, checkpoint_path, history_path
    )

    # Test
    print("\nStarting final evaluation on test set...")
    _, final_test_iou = train_model.run_test_evaluation(model, test_loader, criterion, device, config)
    
    history["final_test_iou"] = final_test_iou
    print(f"Final Test IoU: {final_test_iou:.4f}")

    # Export history and training curves
    with open(history_path, "w") as f:
        json.dump(history, f)
    
    utils.save_training_curves(history, plot_path)
    
    # Removing checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint removed.")

if __name__ == "__main__":
    main()