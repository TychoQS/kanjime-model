import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os 

class ETL9Dataset(torch.utils.data.Dataset):
    """Dataset for ETL9 dataset"""
    def __init__(self, root_dir, transform=None, max_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 
        
        subfolders = sorted([f for f in os.listdir(root_dir) if f.endswith('_unpack')])
        
        if not subfolders:
            print(f"WARNING: No '_unpack' folders found in {root_dir}")
        
        # Getting only the number of selected classes
        if max_classes is not None:
            all_classes = set()
            
            for folder in subfolders:
                folder_path = os.path.join(root_dir, folder)
                csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
                
                if csv_files:
                    csv_path = os.path.join(folder_path, csv_files[0])
                    try:
                        df = pd.read_csv(csv_path)
                        for label_char in df.iloc[:, 1]:
                            # Kanji based on Unicode range
                            if '\u4e00' <= label_char <= '\u9fff':
                                all_classes.add(label_char)
                    except Exception as e:
                        print(f"Error while reading {csv_path}: {e}")
            
            # Select the first max_classes
            all_classes_sorted = sorted(list(all_classes))
            selected_classes = set(all_classes_sorted[:max_classes])
        else:
            selected_classes = None
        
        # Load only images from that classes
        for folder in subfolders:
            folder_path = os.path.join(root_dir, folder)
            csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
            
            if csv_files:
                csv_path = os.path.join(folder_path, csv_files[0])
                try:
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        label_char = row.iloc[1]
                        
                        # Kanji based on Unicode range
                        if '\u4e00' <= label_char <= '\u9fff':
                            if selected_classes is None or label_char in selected_classes:
                                img_name = f"{int(row.iloc[0]):05d}.png"
                                img_path = os.path.join(folder_path, img_name)
                                
                                if os.path.exists(img_path):
                                    self.samples.append((img_path, label_char))
                                    
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
        
        self.classes = sorted(list(set([x[1] for x in self.samples])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Dataset initialized. Images: {len(self.samples)}. Classes (Kanji): {len(self.classes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_char = self.samples[idx]
        image = Image.open(img_path).convert('L')
        label_idx = self.class_to_idx[label_char]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label_idx)