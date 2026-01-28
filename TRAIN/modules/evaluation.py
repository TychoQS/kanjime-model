import os
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from modules import image_processing as ip

def enable_dropout(model):
    """Activates dropout in the model for Monte Carlo Dropout evaluations"""
    for module in model.modules():
        if module.__class__.__name__.find('Dropout') != -1:
            module.train()

def predict_and_evaluate(model, folder_path, class_names, device, img_size, 
                         skip_not_in_classes=False, display=True, mc_iterations=0):
    """Make inferences and show stats with the directory passed as parameter.
    Current admited formats:
        - "class_name.jpg/png/jpeg"
        - "class_name/2332.jpg/png/jpeg"
    Parameters:
        - skip_not_in_classes: If true, images which classes aren't contained in the model's classes are skipped
        - display: If true, display a plot of the image, the binarized image and top 3 predictions
        - mc_iterations: If is greater than 0, apply Monte Carlo Dropout with mc_iterations iterations
    """
    
    num_classes = len(class_names)
    set_classes = set(class_names)
    
    if num_classes < 5:
        print("Not enough classes to evaluate.")
        return

    # Define inference transformations
    inference_transforms = transforms.Compose([
        transforms.ToTensor(),
        ip.NORM
    ])

    # Scan directory for images
    files = [os.path.join(root, f) 
             for root, dirs, files in os.walk(folder_path) 
             for f in files 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # print(files)
    if not files: 
        print("No images found in the folder.")
        return

    if skip_not_in_classes: # Skipping images which class is not contained in the classes of the dataset
        valid_files = []
        for file_name in files:
            # Getting class from filename
            ground_truth = file_name.split("/")[-1][0] 
            if ground_truth.isdigit():
                ground_truth = file_name.split("/")[-2][0] # Getting parent folder name as classs
            if ground_truth in class_names: valid_files.append(file_name)
        files = valid_files

    top1_correct, top3_correct, top5_correct, total_evaluated = 0, 0, 0, 0
    rows = len(files)
    
    if display:
        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(8, 4 * rows))
    if rows == 1: axes = [axes] # Handle single image case

    model.eval() # Set to evaluation mode
    for i, file_name in enumerate(files):
        # Getting class from filename
        ground_truth = file_name.split("/")[-1][0] 
        if ground_truth.isdigit():
            ground_truth = file_name.split("/")[-2][0] # Getting parent folder name as classs
        
        # Load and preprocess
        original_img_show = Image.open(file_name)
        prepared_img_pil = ip.preprocess_image(file_name, img_size)
        img_tensor = inference_transforms(prepared_img_pil).unsqueeze(0).to(device)

        if mc_iterations > 0:
            enable_dropout(model)
            all_probs = []
            with torch.no_grad():
                for _ in range(mc_iterations):
                    outputs = model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    all_probs.append(probs)
            # Meand of probs
            probs = torch.stack(all_probs).mean(dim=0)
            # Computing uncertainty
            # uncertainty = torch.stack(all_probs).std(dim=0)     
            model.eval()   
        else:
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)

        top5_prob, top5_idx = torch.topk(probs, 5)
        top5_indexes = top5_idx[0].cpu().numpy()
        top5_classes = [class_names[idx] for idx in top5_indexes]
        
        # Text color logic and counting
        title_color = 'blue' 
        if ground_truth in set_classes:
            total_evaluated += 1
            # Check Top 1 
            if ground_truth == top5_classes[0]:
                top1_correct += 1
                title_color = 'green' 
            # Check Top 3 
            if ground_truth in top5_classes[:3]:
                top3_correct += 1
                if title_color != 'green': title_color = 'yellow' 
            # Check Top 5
            if ground_truth in top5_classes:
                top5_correct += 1
                if title_color not in ['green', 'yellow']: title_color = 'orange'
            
            if ground_truth not in top5_classes:
                title_color = 'red' 

        if not display: continue # Skip if not displaying

        # Plot original image
        axes[i][0].imshow(original_img_show)
        axes[i][0].set_title("Original Drawing")
        axes[i][0].axis('off')

        # Plot the replication of what model sees
        visible_tensor = ip.denormalize(img_tensor.squeeze(0))
        axes[i][1].imshow(visible_tensor)
        
        # Format results text
        res_txt = ""
        top3_p = top5_prob[0][:3].cpu().numpy()
        top3_i = top5_indexes[:3]
        for idx, p in zip(top3_i, top3_p):
            res_txt += f"{class_names[idx]} : {p*100:.1f}%\n"
            
        axes[i][1].set_title(res_txt, color=title_color, loc='left')
        axes[i][1].axis('off')

    if display:
        plt.tight_layout()
        plt.show()
    
    # Print accuracy stats
    if total_evaluated > 0:
        print(f"--- Predictions Results ({total_evaluated} images) ---")
        print(f"Top-1 Accuracy: {top1_correct}/{total_evaluated} ({top1_correct/total_evaluated*100:.2f}%)")
        print(f"Top-3 Accuracy: {top3_correct}/{total_evaluated} ({top3_correct/total_evaluated*100:.2f}%)")
        print(f"Top-5 Accuracy: {top5_correct}/{total_evaluated} ({top5_correct/total_evaluated*100:.2f}%)")
    else:
        print("Accuracy not evaluated (filenames do not match known classes)")