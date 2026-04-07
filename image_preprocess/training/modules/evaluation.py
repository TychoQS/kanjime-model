import torch

def calculate_iou(preds, masks, threshold=0.5):
    # Intersection over Union
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    
    if union == 0:
        return 1.0
    return (intersection / union).item()