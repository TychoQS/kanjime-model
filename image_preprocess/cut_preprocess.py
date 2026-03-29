import sys
import os
import cv2
import numpy as np
import torch
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.preprocess_utils import ImageOutputSaver

# CUT model path assumptions
CUT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "output", "cut_training")
CUT_MODEL_NAME = "kanji_cut_128"
CUT_GENERATOR_PATH = os.path.join(CUT_CHECKPOINT_DIR, CUT_MODEL_NAME, "latest_net_G.pth")
IMG_SIZE = 128

def load_cut_generator(device):
    """Load the CUT generator network for image-to-image translation."""
    cut_dir = os.path.join(SCRIPT_DIR, "models", "CUT")
    original_path = sys.path.copy()
    sys.path.insert(0, cut_dir)
    try:
        from models.networks import define_G
        netG = define_G(
            input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks',
            norm='instance', use_dropout=False, init_type='xavier',
            init_gain=0.02, no_antialias=False, no_antialias_up=False,
            gpu_ids=[], opt=None
        )
        state_dict = torch.load(CUT_GENERATOR_PATH, map_location=device)
        netG.load_state_dict(state_dict)
        netG = netG.to(device)
        netG.eval()
        return netG
    finally:
        sys.path = original_path

def cut_preprocess(img_bgr, netG, device):
    """
    Apply CUT generator to transform scene text to clean handwriting domain.
    Returns:
        binary: Binary image (uint8, white text on black background)
        steps: List of (image, label) tuples for visualization
    """
    original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    steps = [
        (original, "1. Original")
    ]

    # Normalize to [-1, 1] (CUT standard)
    img_tensor = transforms.ToTensor()(img_resized)
    img_tensor = (img_tensor - 0.5) / 0.5
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = netG(img_tensor)

    # Denormalize [-1,1] -> [0, 255]
    output = output.squeeze(0).cpu()
    output = ((output + 1) / 2.0).clamp(0, 1)
    output_np = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    steps.append((output_np.copy(), "2. CUT result"))

    return output_np, steps

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cut_preprocess.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CUT generator on {device}...")
    netG = load_cut_generator(device)

    binary, steps = cut_preprocess(img, netG, device)

    filename = os.path.basename(img_path)
    script_name = "cut_preprocess"

    saver = ImageOutputSaver("output")
    saver.save_mosaic(steps, filename, script_name)
    print(f"Saved results for {filename}")
