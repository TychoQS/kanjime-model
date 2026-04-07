import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

class SyntheticKanjiDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root = Path(root_dir) / split
        self.images_path = sorted(list((self.root / "images").glob("*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        mask_path = img_path.parent.parent / "masks" / f"{img_path.stem}_mask.png"
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        mask_tensor = transforms.ToTensor()(mask)
        mask_tensor = (mask_tensor > 0.5).float()
            
        return image, mask_tensor