import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Setting paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.preprocess_utils import ImageOutputSaver

# Model path
NST_MODEL_DIR = os.path.join(SCRIPT_DIR, "output", "pytorch_examples_nst_training")
NST_MODEL_PATH = os.path.join(NST_MODEL_DIR, "jspc1400_trained_nst.model")

def load_nst_model(model_path, device):
    """Load the trained NST transformer network importing it from models."""
    nst_lib_path = os.path.join(SCRIPT_DIR, "models", "example_pytorch", "fast_neural_style")
    if nst_lib_path not in sys.path:
        sys.path.insert(0, nst_lib_path)
    
    try:
        from neural_style.transformer_net import TransformerNet
        
        model = TransformerNet()
        state_dict = torch.load(model_path, map_location=device)
        
        # Consistent key cleaning
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        return model
    except ImportError as e:
        print(f"Error: Could not import TransformerNet from {nst_lib_path}. Details: {e}")
        sys.exit(1)

def nst_preprocess(img_bgr, model, device):
    """
    Apply Neural Style Transfer to an image.
    Returns:
        binary: Binary image (uint8, white text on black background)
        steps: List of (image, label) tuples for visualization
    """
    original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocessing transforms (Standard for PyTorch NST example)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    content_image = Image.fromarray(original)
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(content_image).cpu()
    
    # Postprocessing
    output = output.squeeze(0).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    
    steps = [
        (original, "1. Original"),
        (output.copy(), "2. NST Output")
    ]
    
    return output, steps

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nst_preprocess.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading NST model on {device}...")
    
    if not os.path.exists(NST_MODEL_PATH):
        print(f"Error: Model not found at {NST_MODEL_PATH}")
        sys.exit(1)
        
    model = load_nst_model(NST_MODEL_PATH, device)
    
    binary, steps = nst_preprocess(img, model, device)
    
    filename = os.path.basename(img_path)
    script_name = "nst_preprocess"
    
    saver = ImageOutputSaver("output")
    saver.save_mosaic(steps, filename, script_name)
    print(f"Saved results for {filename}")
