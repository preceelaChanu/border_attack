import argparse
import os
import torch
import glob
import json
import random
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from models import TargetModel
from utils import get_preprocessing_transforms, normalize

def get_args():
    parser = argparse.ArgumentParser(description="Paper Replication Preprocessing")
    parser.add_argument('--data_dir', type=str, required=True, help='Folder with raw images (.jpg, .png)')
    parser.add_argument('--val_map', type=str, default=None, help='Path to .txt file mapping filename to class index (e.g., "img1.jpg 10")')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save .pt tensors and metadata')
    parser.add_argument('--sample_size', type=int, default=5000, help='Initial random sample size (Paper used 5000)')
    return parser.parse_args()

def load_val_map(path):
    """
    Parses a file like:
    ILSVRC2012_val_00000001.JPEG 65
    ILSVRC2012_val_00000002.JPEG 970
    """
    mapping = {}
    if not path:
        return None
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                label = int(parts[1])
                mapping[filename] = label
    return mapping

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure output dir exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Gather Images
    all_images = glob.glob(os.path.join(args.data_dir, '*'))
    all_images = [p for p in all_images if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly sample 5000 as per paper (if we have enough)
    if len(all_images) > args.sample_size:
        print(f"Sampling {args.sample_size} images from {len(all_images)} total...")
        selected_paths = random.sample(all_images, args.sample_size)
    else:
        selected_paths = all_images

    # Load Labels
    val_map = load_val_map(args.val_map)
    
    # Load Model for filtering
    print("Loading ResNet50 for filtering...")
    model = TargetModel(device)
    
    transform = get_preprocessing_transforms() # Resize -> ToTensor
    
    valid_data_info = []
    skipped_gray = 0
    skipped_wrong = 0
    skipped_no_label = 0

    print("Starting filtering process...")
    for img_path in tqdm(selected_paths):
        filename = os.path.basename(img_path)
        
        # Get Ground Truth
        ground_truth = None
        if val_map and filename in val_map:
            ground_truth = val_map[filename]
        else:
            # Fallback: Try to guess from folder name if structured as /n01440764/
            # (Note: converting folder synset ID to int requires a mapping, skipping for brevity)
            # If we don't have a label, we can't filter by "wrongly classified".
            # For this script, we skip if we don't have a label.
            skipped_no_label += 1
            continue

        try:
            # Open Image
            pil_img = Image.open(img_path)
            
            # Check Grayscale (Paper Step 1)
            if pil_img.mode != 'RGB':
                # Some images are 'L' (grayscale) or 'CMYK'
                # Even if we convert to RGB, the paper likely excluded native grayscale
                # to test color style transfer robustness.
                pil_img = pil_img.convert('RGB')
                # Optional: strictly skip if original was not RGB? 
                # Paper says "removing gray images". Let's check channels.
                # If inputs are 1-channel, we skip.
                if len(pil_img.getbands()) < 3:
                     skipped_gray += 1
                     continue

            # Preprocess
            # Resize 224x224 and scale 0-1
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            
            # Check Classification (Paper Step 2)
            with torch.no_grad():
                logits = model(img_tensor) # Model handles normalization internally
                pred = torch.argmax(logits, dim=1).item()
            
            if pred != ground_truth:
                skipped_wrong += 1
                continue
            
            # If we are here, image is Clean, RGB, and Correctly Classified.
            
            # Save the TENSOR (to avoid re-preprocessing float errors)
            save_name = f"{os.path.splitext(filename)[0]}.pt"
            save_path = os.path.join(args.output_dir, save_name)
            torch.save(img_tensor.cpu(), save_path)
            
            valid_data_info.append({
                "filename": save_name,
                "original_filename": filename,
                "label": ground_truth
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Save Metadata
    meta_path = os.path.join(args.output_dir, "dataset_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(valid_data_info, f, indent=4)

    print("\n" + "="*30)
    print("Preprocessing Complete")
    print(f"Total processed: {len(selected_paths)}")
    print(f"Skipped (No Label): {skipped_no_label}")
    print(f"Skipped (Grayscale): {skipped_gray}")
    print(f"Skipped (Wrongly Classified): {skipped_wrong}")
    print(f"Valid Images Saved: {len(valid_data_info)}")
    print(f"Data saved to: {args.output_dir}")
    print("="*30)