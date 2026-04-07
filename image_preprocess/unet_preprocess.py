import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.preprocess_utils import ImageOutputSaver

# Model path
UNET_MODEL_DIR = os.path.join("output", "u-net_training")
UNET_MODEL_PATH = os.path.join(UNET_MODEL_DIR, "best_model.pth")

def load_unet_model(model_path, device):
    """Load the trained U-Net model from the training module."""
    unet_training_dir = os.path.join("training")
    if unet_training_dir not in sys.path:
        sys.path.insert(0, unet_training_dir)
        
    try:
        from modules.architecture import get_unet_model
        
        model = get_unet_model()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except ImportError as e:
        print(f"Error: Could not import get_unet_model from {unet_training_dir}. Details: {e}")
        sys.exit(1)

def unet_preprocess(img_bgr, model, device):
    """
    Apply UNet segmentation to an image.
    """
    original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    h, w = original.shape[:2]
    # Ensure dimensions are divisible by 32 for MobileNetV3 encoder
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    if new_h == 0: new_h = 32
    if new_w == 0: new_w = 32
    
    resized = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Preprocessing
    content_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    input_tensor = content_transform(resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        # Logits to probabilities with sigmoid function
        prob = torch.sigmoid(output)
    
    # Getting output mask
    mask = prob.squeeze().cpu().numpy()
    binary = (mask > 0.5).astype(np.uint8) * 255
    
    # Resize back to original dimensions
    binary_original_size = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    
    steps = [
        (original, "1. Original"),
        (binary_original_size.copy(), "2. UNet Mask")
    ]
    
    return binary_original_size, steps

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python unet_preprocess.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading UNet model on {device}...")
    
    if not os.path.exists(UNET_MODEL_PATH):
        print(f"Error: Model not found at {UNET_MODEL_PATH}")
        sys.exit(1)
        
    model = load_unet_model(UNET_MODEL_PATH, device)
    
    binary, steps = unet_preprocess(img, model, device)
    
    filename = os.path.basename(img_path)
    script_name = "unet_preprocess"
    
    saver = ImageOutputSaver("output")
    saver.save_mosaic(steps, filename, script_name)
    print(f"Saved results for {filename}")
