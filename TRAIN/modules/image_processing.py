import cv2
import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms

NORM = transforms.Compose([])

def binarize(img):
    """Applies binarization"""
    img_array = np.array(img)
    _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(img_array)


def preprocess_image(image_path, img_size, num_channels=1):
    """Process the image to be as equal as possible as the input images of the model"""
    img = Image.open(image_path).convert('L')
    
    # Detecting if the background is more black or white
    img_array = np.array(img)
    mean_intensity = img_array.mean()
    
    if mean_intensity > 127.5:  # White background --> Invert
        img = ImageOps.invert(img)
    
    # Resize and convert to channels
    img = img.resize((img_size, img_size))
    img = img.convert('RGB') if num_channels == 3 else img.convert('L')

    # Binarize
    img = binarize(img)
    
    return img

def denormalize(tensor):
    """Reverts normalization to display the image correctly"""
    img = tensor.clone().detach().cpu()
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()