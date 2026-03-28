import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.exposure import is_low_contrast
from skimage.restoration import estimate_sigma
from skimage.filters import threshold_niblack, threshold_sauvola
from utils.preprocess_utils import ImageOutputSaver

img = cv2.imread(sys.argv[1])
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

method = sys.argv[2].lower() if len(sys.argv) > 2 else "otsu"
assert method in ["otsu", "niblack", "sauvola"], "Method must be 'otsu', 'niblack' or 'sauvola'"

use_nlm = "-nlm" in [arg.lower() for arg in sys.argv]

steps = [
    (original, "1. Original"),
    (gray,     "2. Grayscale"),
]

applied_clahe = False
if is_low_contrast(img_as_float(gray)):
    clahe = cv2.createCLAHE(clipLimit=3)
    gray = clahe.apply(gray)
    applied_clahe = True
    steps.append((gray.copy(), "3. CLAHE"))

H, W = gray.shape
s = estimate_sigma(gray) * 255

if use_nlm:
    gray = cv2.fastNlMeansDenoising(gray, h=s)
    denoise_label = "FastNLMeansDenoising"
    suffix = "_nlmeansdenoising"
else:
    sigma_color = s
    sigma_space = max(5, int(min(H, W) * 0.01))
    gray = cv2.bilateralFilter(gray, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    denoise_label = "BilateralFilter"
    suffix = "_bilateral"

steps.append((gray.copy(), f"{'4' if applied_clahe else '3'}. {denoise_label}"))

if method == "otsu":
    _, thresholded_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
elif method == "niblack":
    window_size = int(min(H, W) * 0.2)
    if window_size % 2 == 0:
        window_size += 1
    thresh_n = threshold_niblack(gray, window_size=window_size, k=0.2)
    thresholded_img = ((gray > thresh_n).astype(np.uint8) * 255)
    thresholded_img = 255 - thresholded_img
else:
    window_size = int(min(H, W) * 0.2)
    if window_size % 2 == 0:
        window_size += 1
    thresh_s = threshold_sauvola(gray, window_size=window_size, k=0.2)
    thresholded_img = ((gray > thresh_s).astype(np.uint8) * 255)
    thresholded_img = 255 - thresholded_img

steps.append((thresholded_img.copy(), f"{'5' if applied_clahe else '4'}. {method.capitalize()}"))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
result = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel)
steps.append((result.copy(), f"{'6' if applied_clahe else '5'}. Close"))

fig, axes = plt.subplots(1, len(steps), figsize=(3 * len(steps), 4))
for i, (image, title) in enumerate(steps):
    axes[i].imshow(image, cmap=None if image.ndim == 3 else "gray")
    axes[i].set_title(title, fontsize=9)
    axes[i].axis("off")

plt.tight_layout()

filename = os.path.basename(sys.argv[1])
name_base = os.path.splitext(os.path.basename(sys.argv[0]))[0]
script_name = f"{name_base}_{method}{suffix}"

saver = ImageOutputSaver("output")
saver.save_mosaic(steps, filename, script_name)