import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image

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