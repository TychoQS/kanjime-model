#!/usr/bin/env python3
"""
Benchmark script for evaluating image preprocessing pipelines.

Loads Kanji-range images from the JPSC1400 dataset, applies different
preprocessing pipelines, and measures Top-5 accuracy using the trained
multi-head kanji classification model.

Results are saved to stats/image_preprocess_benchmark.csv
"""

import os
import sys
import csv
import json
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from custom_binarization import custom_binarize
from milyaev import milyaev_binarize
from cut_preprocess import cut_preprocess as cut_preprocess_fn, load_cut_generator
from nst_preprocess import nst_preprocess as nst_preprocess_fn, load_nst_model
from unet_preprocess import unet_preprocess as unet_preprocess_fn, load_unet_model, UNET_MODEL_PATH

# Setting paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Root of the project

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# JPSC1400 dataset 
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "JPSC1400-20201218")
LABEL_FILE = os.path.join(DATASET_DIR, "label.txt")
IMAGE_DIR = os.path.join(DATASET_DIR, "png")

# Kanji classification model 
TRAINING_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training", "training_output")
CLASSES_PATH = os.path.join(TRAINING_OUTPUT_DIR, "classes.json")
RADICAL_CLASSES_PATH = os.path.join(TRAINING_OUTPUT_DIR, "radical_classes.json")
STROKE_CLASSES_PATH = os.path.join(TRAINING_OUTPUT_DIR, "stroke_classes.json")
MODEL_PATH = os.path.join(TRAINING_OUTPUT_DIR, "best_kanji_model.pth")

# CUT model
CUT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "output", "cut_training")
CUT_MODEL_NAME = "kanji_cut_128"
CUT_GENERATOR_PATH = os.path.join(CUT_CHECKPOINT_DIR, CUT_MODEL_NAME, "latest_net_G.pth")

# NST model
NST_MODEL_DIR = os.path.join(SCRIPT_DIR, "output", "pytorch_examples_nst_training")
NST_MODEL_PATH = os.path.join(NST_MODEL_DIR, "jspc1400_trained_nst.model")

# Outputs paths
STATS_DIR = os.path.join(PROJECT_ROOT, "stats")
OUTPUT_CSV = os.path.join(STATS_DIR, "image_preprocess_benchmark.csv")

# Kanji Unicode range
CJK_UNIFIED_START = '\u4e00'
CJK_UNIFIED_END = '\u9fff'

# Model config
IMG_SIZE = 128
CHANNEL_SIZE = 3


# Data loading
def load_kanji_dataset():
    """Load images and ground truth from JPSC1400, filtered to Kanji the range defined by the limits."""
    entries = []
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            img_id = parts[0]
            char = parts[2]
            if CJK_UNIFIED_START <= char <= CJK_UNIFIED_END:
                img_path = os.path.join(IMAGE_DIR, f"{img_id}.png")
                if os.path.exists(img_path):
                    entries.append((img_path, char))
    return entries


def load_model_classes():
    """Load model class definitions from training output."""
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    with open(RADICAL_CLASSES_PATH, "r", encoding="utf-8") as f:
        radical_classes = json.load(f)
    with open(STROKE_CLASSES_PATH, "r", encoding="utf-8") as f:
        stroke_classes = json.load(f)
    return classes, radical_classes, stroke_classes


def load_kanji_model(classes, radical_classes, stroke_classes, device):
    """Load the trained multi-head kanji classification model."""
    training_dir = os.path.join(PROJECT_ROOT, "training")
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    from modules.models import build_multi_head_model

    model = build_multi_head_model(
        num_kanji_classes=len(classes),
        num_radical_classes=len(radical_classes),
        num_stroke_classes=len(stroke_classes)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Inference
def prepare_for_inference(processed_gray_img):
    """Convert a processed grayscale image to a tensor for model inference.
    Mirrors val_transforms: Resize -> Binarize (Otsu) -> Grayscale(3ch) -> ToTensor
    """
    img_resized = cv2.resize(processed_gray_img, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_NEAREST)

    # Ensure white-text-on-black-bg orientation
    if np.mean(img_resized) > 127.5:
        img_resized = 255 - img_resized

    pil_img = Image.fromarray(img_resized).convert('RGB')
    tensor = transforms.ToTensor()(pil_img)
    return tensor.unsqueeze(0)


def evaluate_inference(model, img_tensor, true_class_idx, device):
    """Evaluate inference and return (is_top5_correct, max_logit)"""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        kanji_logits, _, _ = model(img_tensor)
        max_logit = torch.max(kanji_logits).item()
        _, top5_idx = torch.topk(kanji_logits, 5)
        top5_indices = top5_idx[0].cpu().numpy()
    return int(true_class_idx) in top5_indices, max_logit


# Preprocessing pipelines
def baseline_otsu(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() > 127.5:
        gray = 255 - gray
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def cb_otsu_bilateral(img_bgr):
    result, _, _ = custom_binarize(img_bgr, method="otsu", use_nlm=False)
    return result

def cb_otsu_nlm(img_bgr):
    result, _, _ = custom_binarize(img_bgr, method="otsu", use_nlm=True)
    return result

def cb_niblack_bilateral(img_bgr):
    result, _, _ = custom_binarize(img_bgr, method="niblack", use_nlm=False)
    return result

def cb_niblack_nlm(img_bgr):
    result, _, _ = custom_binarize(img_bgr, method="niblack", use_nlm=True)
    return result

def cb_sauvola_bilateral(img_bgr):
    result, _, _ = custom_binarize(img_bgr, method="sauvola", use_nlm=False)
    return result

def cb_sauvola_nlm(img_bgr):
    result, _, _ = custom_binarize(img_bgr, method="sauvola", use_nlm=True)
    return result

def milyaev_bilateral(img_bgr):
    result_normal, result_inverted, _ = milyaev_binarize(img_bgr, use_nlm=False)
    return result_normal, result_inverted

def milyaev_nlm(img_bgr):
    result_normal, result_inverted, _ = milyaev_binarize(img_bgr, use_nlm=True)
    return result_normal, result_inverted


def get_pipelines(device):
    """Returns a dictionary of {name: function} with all the preprocessing pipelines.
    """
    pipelines = {
        "baseline_otsu": baseline_otsu,
        "custom_binarization_otsu_bilateral": cb_otsu_bilateral,
        "custom_binarization_otsu_nlm": cb_otsu_nlm,
        "custom_binarization_niblack_bilateral": cb_niblack_bilateral,
        "custom_binarization_niblack_nlm": cb_niblack_nlm,
        "custom_binarization_sauvola_bilateral": cb_sauvola_bilateral,
        "custom_binarization_sauvola_nlm": cb_sauvola_nlm,
        "milyaev_bilateral": milyaev_bilateral,
        "milyaev_nlm": milyaev_nlm,
    }

    try:
        netG = load_cut_generator(device)
        pipelines["cut_model"] = lambda img: cut_preprocess_fn(img, netG, device)[0]
    except Exception as e:
        print(f"WARNING: Could not load CUT generator, skipping: {e}")

    try:
        if os.path.exists(NST_MODEL_PATH):
            nst_model = load_nst_model(NST_MODEL_PATH, device)
            pipelines["nst_model"] = lambda img: nst_preprocess_fn(img, nst_model, device)[0]
        else:
            print(f"WARNING: NST model not found at {NST_MODEL_PATH}, skipping.")
    except Exception as e:
        print(f"WARNING: Could not load NST model, skipping: {e}")

    try:
        if os.path.exists(UNET_MODEL_PATH):
            unet_model = load_unet_model(UNET_MODEL_PATH, device)
            pipelines["unet_model"] = lambda img: unet_preprocess_fn(img, unet_model, device)[0]
        else:
            print(f"WARNING: UNet model not found at {UNET_MODEL_PATH}, skipping.")
    except Exception as e:
        print(f"WARNING: Could not load UNet model, skipping: {e}")

    return pipelines


# Benchmark execution
def run_benchmark(pipelines, model, valid_entries, class_to_idx, device):
    """Run all pipelines and return results."""
    results = []

    for name, pipeline_fn in pipelines.items():
        print(f"\n--- Pipeline: {name} ---")
        top5_count = 0
        errors = 0

        for img_path, true_char in tqdm(valid_entries, desc=name):
            try:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    errors += 1
                    continue
                
                processed = pipeline_fn(img_bgr)
                true_idx = class_to_idx[true_char]

                # Handles methods returning multiple candidates (e.g. Milyaev normal/inverted)
                if isinstance(processed, (list, tuple)):
                    candidates = processed
                else:
                    candidates = [processed]

                best_is_top5 = False
                best_confidence = -float('inf')

                for candidate_img in candidates:
                    tensor = prepare_for_inference(candidate_img)
                    is_top5, confidence = evaluate_inference(model, tensor, true_idx, device)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_is_top5 = is_top5

                if best_is_top5:
                    top5_count += 1

            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Error on {os.path.basename(img_path)}: {e}")

        total = len(valid_entries)
        pct = top5_count / total * 100 if total > 0 else 0
        print(f"  Top-5: {top5_count}/{total} ({pct:.2f}%)")
        if errors > 0:
            print(f"  Errors: {errors}")
        results.append((name, top5_count, total, f"{pct:.2f}"))

    return results


def save_results(results):
    """Write benchmark results to CSV."""
    os.makedirs(STATS_DIR, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline_name", "top5_count", "total", "percentage"])
        writer.writerows(results)
    print(f"\nResults saved to: {OUTPUT_CSV}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset (Kanji range only)
    kanji_entries = load_kanji_dataset()
    print(f"Loaded {len(kanji_entries)} Kanji-range images from JPSC1400")

    # Load model class definitions
    classes, radical_classes, stroke_classes = load_model_classes()
    print(f"Model: {len(classes)} kanji, {len(radical_classes)} radicals, "
          f"{len(stroke_classes)} strokes")

    # Class coverage analysis
    dataset_chars = set(char for _, char in kanji_entries)
    model_chars = set(classes)
    covered = dataset_chars & model_chars
    covered_by_dataset = len(covered)/len(model_chars)
    
    print(f"\nKanjis in dataset:  {len(dataset_chars)}")
    print(f"Kanji classes in model:   {len(model_chars)}")
    print(f"Covered by model:         {len(covered)}/{len(dataset_chars)} "
          f"({len(covered) / len(dataset_chars) * 100:.1f}%)")
    print(f"Covered by dataset:   {len(covered)}/{len(model_chars)} ({covered_by_dataset:.1f}%)")
    not_covered = model_chars - dataset_chars
    if not_covered:
        print(f"Not covered in dataset ({len(not_covered)}): "
              f"{''.join(sorted(not_covered)[:20])}"
              f"{'...' if len(not_covered) > 20 else ''}")

    # Filter to entries whose class exists in the model
    class_to_idx = {c: i for i, c in enumerate(classes)}
    valid_entries = [(p, c) for p, c in kanji_entries if c in class_to_idx]
    print(f"Valid entries for benchmark (Nº images): {len(valid_entries)}")

    if not valid_entries:
        print("ERROR: No valid entries found. Exiting.")
        return

    # Load kanji classification model
    model = load_kanji_model(classes, radical_classes, stroke_classes, device)

    # Build pipelines and run benchmark
    pipelines = get_pipelines(device)
    print(f"\nRunning benchmark ({len(pipelines)} pipelines)...")
    results = run_benchmark(pipelines, model, valid_entries, class_to_idx, device)

    # Save results
    save_results(results)

    # Summary
    print(f"\n{'Pipeline':<45} {'Top-5':>8} {'Total':>6} {'%':>8}")
    print("-" * 70)
    for name, top5, total, pct in results:
        print(f"{name:<45} {top5:>8} {total:>6} {pct:>7}%")


if __name__ == "__main__":
    main()