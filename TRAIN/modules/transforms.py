import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image
from modules import image_processing as ip

class GaussianNoise(object):
    """Adds random noise to the image to improve model robustness"""
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class MorphologicalTransform(object):
    """Applies randomly Erode or Dilate operations"""
    def __init__(self, kernel_size=3, p=0.5):
        self.kernel_size = kernel_size
        self.p = p
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    def __call__(self, img):
        img_np = np.array(img)
        
        if np.random.random() < self.p:
            operation = np.random.choice(['erode', 'dilate'])
            if operation == 'erode':
                img_np = cv2.erode(img_np, self.kernel, iterations=1)
            else:
                img_np = cv2.dilate(img_np, self.kernel, iterations=1)
        return Image.fromarray(img_np)

def get(img_size, channel_size):
    """Returns the training and validation transforms"""
    train_transforms = transforms.Compose([ 
        transforms.Resize((img_size, img_size)), 
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5), 
        transforms.RandomAffine( 
            degrees=20,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=5
        ),
        transforms.ElasticTransform(alpha=10.0, sigma=5.0),     
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), 
        MorphologicalTransform(kernel_size=5, p=0.5),
        transforms.Lambda(lambda x: ip.binarize(x)),
        transforms.Grayscale(num_output_channels=channel_size),
        transforms.ToTensor(),
        # GaussianNoise(0., 0.03), 
        transforms.RandomErasing(p=0.3), 
        ip.NORM
    ])

    val_transforms = transforms.Compose([ 
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda x: ip.binarize(x)),
        transforms.Grayscale(num_output_channels=channel_size),
        transforms.ToTensor(),
        ip.NORM
    ])

    return train_transforms, val_transforms
