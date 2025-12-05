import argparse
import os
import torch
import json
from tqdm import tqdm
from models import TargetModel, Vgg19Fidelity
from attack import StealthyBorderAttack
from utils import load_tensor_data, save_image

def get_args():
    parser = argparse.ArgumentParser(description="Stealthy Border Attack - Main Phase")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing processed .pt files')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save adversarial images')
    parser.add_argument('--attack_type', type=str, choices=['targeted', 'untargeted'], required=True)
    parser.add_argument('--target_class', type=int, default=None, help='Target class index (required for targeted)')
    parser.add_argument('--border_width', type=int, default=4, help='Width of the adversarial border')
    parser.add_argument('--max_iter', type=int, default=50, help='Maximum optimization iterations')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.attack_type == 'targeted' and args.target_class is None:
        raise ValueError("Target class must be specified for targeted attack")
        
    # Load Metadata created by preprocess.py
    meta_path = os.path.join(args.data_dir, "dataset_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"dataset_meta.json not found in {args.data_dir}. Run preprocess.py first.")
        
    with open(meta_path, 'r') as f:
        dataset_meta = json.load(f)

    # Initialize models
    print("Loading models...")
    target_model = TargetModel(device)
    fidelity_model = Vgg19Fidelity(device)
    
    # Initialize Attacker
    attacker = StealthyBorderAttack(
        target_model, 
        fidelity_model, 
        device,
        border_width=args.border_width,
        max_iter=args.max_iter,
        lambda_a=1.0 if args.attack_type == 'targeted' else 1.5,
        lambda_f=1000.0
    )
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    success_count = 0
    total_images = 0
    
    print(f"Starting {args.attack_type} attack on {len(dataset_meta)} valid images...")
    
    for entry in tqdm(dataset_meta):
        filename = entry['filename']      # e.g., "img1.pt"
        gt_label = entry['label']         # Actual class index
        
        file_path = os.path.join(args.data_dir, filename)
        
        try:
            # Load preprocessed tensor directly
            image = load_tensor_data(file_path, device)
            
            # Determine target for the attack
            if args.attack_type == 'targeted':
                target = args.target_class
                # Skip if the image is ALREADY the target class (rare but possible)
                if gt_label == target:
                    continue
            else:
                target = gt_label # For untargeted, we pass GT so the loss knows what to avoid
            
            # Run Attack
            adv_image, is_success = attacker.run(image, args.attack_type, target)
            
            if is_success:
                success_count += 1
            
            total_images += 1
            
            # Save the resulting adversarial image as PNG (for viewing)
            save_name = f"adv_{os.path.splitext(filename)[0]}.png"
            save_path = os.path.join(args.output_dir, save_name)
            save_image(adv_image, save_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if total_images > 0:
        asr = (success_count / total_images) * 100.0
        print("\n" + "="*30)
        print(f"Attack Type: {args.attack_type}")
        print(f"Total Valid Images Attacked: {total_images}")
        print(f"Successful Attacks: {success_count}")
        print(f"Attack Success Rate (ASR): {asr:.2f}%")
        print("="*30)
    else:
        print("No images were attacked.")

if __name__ == "__main__":
    main()