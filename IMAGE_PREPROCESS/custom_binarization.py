import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float, img_as_ubyte
from skimage.exposure import is_low_contrast
from skimage.restoration import denoise_nl_means, estimate_sigma
from preprocess_utils import ImageOutputSaver

img = cv2.imread(sys.argv[1])
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

sigma = estimate_sigma(img_as_float(gray))
sigmaColor = sigma * 255
sigmaSpace = max(5, int(min(gray.shape) * 0.01))
gray = cv2.bilateralFilter(gray, d=9, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
steps.append((gray.copy(), f"{'4' if applied_clahe else '3'}. Bilateral Filter"))

_, thresholded_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
steps.append((thresholded_img.copy(), f"{'5' if applied_clahe else '4'}. Otsu"))

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
saver = ImageOutputSaver("output")
saver.save_figure(filename, script_name)