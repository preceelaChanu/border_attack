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
    print(f"Loading val map from: {path}")
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                label = int(parts[1])
                mapping[filename] = label
    print(f"Loaded {len(mapping)} entries from val map")
    return mapping

def main():
    print("Starting preprocessing...")
    args = get_args()
    print(f"Arguments: {args}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ensure output dir exists
    print(f"Creating output directory: {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"Output directory created/verified")

    # 1. Gather Images
    print(f"Gathering images from: {args.data_dir}")
    all_images = glob.glob(os.path.join(args.data_dir, '*'))
    all_images = [p for p in all_images if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(all_images)} images")
    
    if len(all_images) == 0:
        print("No images found! Exiting.")
        return
    
    # Randomly sample 5000 as per paper (if we have enough)
    if len(all_images) > args.sample_size:
        print(f"Sampling {args.sample_size} images from {len(all_images)} total...")
        selected_paths = random.sample(all_images, args.sample_size)
    else:
        selected_paths = all_images
    print(f"Selected {len(selected_paths)} images for processing")

    # Load Labels
    val_map = load_val_map(args.val_map)
    if not val_map:
        print("No val_map provided or failed to load. Exiting.")
        return
    
    # Load Model for filtering
    print("Loading ResNet50 for filtering...")
    try:
        model = TargetModel(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    transform = get_preprocessing_transforms() # Resize -> ToTensor
    print("Transform pipeline created")
    
    valid_data_info = []
    skipped_gray = 0
    skipped_wrong = 0
    skipped_no_label = 0

    print("Starting filtering process...")
    for i, img_path in enumerate(tqdm(selected_paths)):
        print(f"Processing {i+1}/{len(selected_paths)}: {img_path}")
        filename = os.path.basename(img_path)
        
        # Get Ground Truth
        ground_truth = None
        if val_map and filename in val_map:
            ground_truth = val_map[filename]
            print(f"  Ground truth label: {ground_truth}")
        else:
            print(f"  No label found for {filename}, skipping")
            skipped_no_label += 1
            continue

        try:
            # Open Image
            print(f"  Opening image...")
            pil_img = Image.open(img_path)
            print(f"  Image mode: {pil_img.mode}, size: {pil_img.size}")
            
            # Check Grayscale (Paper Step 1)
            if pil_img.mode != 'RGB':
                print(f"  Converting from {pil_img.mode} to RGB")
                pil_img = pil_img.convert('RGB')
                if len(pil_img.getbands()) < 3:
                     print(f"  Skipping grayscale image")
                     skipped_gray += 1
                     continue

            # Preprocess
            print(f"  Preprocessing image...")
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            print(f"  Tensor shape: {img_tensor.shape}")
            
            # Check Classification (Paper Step 2)
            print(f"  Running model inference...")
            with torch.no_grad():
                logits = model(img_tensor)
                pred = torch.argmax(logits, dim=1).item()
            print(f"  Predicted: {pred}, Ground truth: {ground_truth}")
            
            if pred != ground_truth:
                print(f"  Prediction mismatch, skipping")
                skipped_wrong += 1
                continue
            
            print(f"  Image passed all filters!")
            
            # Save the TENSOR (to avoid re-preprocessing float errors)
            save_name = f"{os.path.splitext(filename)[0]}.pt"
            save_path = os.path.join(args.output_dir, save_name)
            torch.save(img_tensor.cpu(), save_path)
            print(f"  Saved to: {save_path}")
            
            valid_data_info.append({
                "filename": save_name,
                "original_filename": filename,
                "label": ground_truth
            })
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save Metadata
    meta_path = os.path.join(args.output_dir, "dataset_meta.json")
    print(f"Saving metadata to: {meta_path}")
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

if __name__ == "__main__":
    main()