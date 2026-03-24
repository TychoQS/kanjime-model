import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float, img_as_ubyte
from skimage.exposure import is_low_contrast
from skimage.restoration import estimate_sigma
from skimage.filters import threshold_niblack, threshold_sauvola
from preprocess_utils import ImageOutputSaver

img = cv2.imread(sys.argv[1])
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

method = sys.argv[2].lower() if len(sys.argv) > 2 else "otsu"
assert method in ["otsu", "niblack", "sauvola"], "Method must be 'otsu', 'niblack' or 'sauvola'"

steps = [
    (original, "1. Original"),
    (gray,     "2. Grayscale"),
]

applied_clahe = False
if is_low_contrast(img_as_float(gray)):
    clahe = cv2.createCLAHE(clipLimit=3)
    gray = clahe.apply(gray)
    applied_clahe = True
    steps.append((gray.copy(), f"3. CLAHE"))

sigma = estimate_sigma(img_as_float(gray))
sigmaColor = sigma * 255
sigmaSpace = max(5, int(min(gray.shape) * 0.01))
gray = cv2.bilateralFilter(gray, d=9, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
steps.append((gray.copy(), f"{'4' if applied_clahe else '3'}. Bilateral Filter"))

if method == "otsu":
    _, thresholded_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
elif method == "niblack":
    H, W = gray.shape
    window_size = int(min(H, W) * 0.4)
    if window_size % 2 == 0:
        window_size += 1
    thresh_n = threshold_niblack(gray, window_size=window_size, k=0.2)
    thresholded_img = ((gray > thresh_n).astype(np.uint8) * 255)
    thresholded_img = 255 - thresholded_img
else:
    H, W = gray.shape
    window_size = int(min(H, W) * 0.4)
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
for ax, (image, title) in zip(axes, steps):
    ax.imshow(image, cmap=None if image.ndim == 3 else "gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.tight_layout()

filename = os.path.basename(sys.argv[1])
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
script_name += f"_{method}"
saver = ImageOutputSaver("output")
saver.save_mosaic(steps, filename, script_name)