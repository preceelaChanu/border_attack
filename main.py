import argparse
import os
import torch
import glob
from tqdm import tqdm
from models import TargetModel, Vgg19Fidelity
from attack import StealthyBorderAttack
from utils import load_image, save_image

def get_args():
    parser = argparse.ArgumentParser(description="Stealthy Border Attack Implementation")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing clean images')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save adversarial images')
    parser.add_argument('--attack_type', type=str, choices=['targeted', 'untargeted'], required=True)
    parser.add_argument('--target_class', type=int, default=None, help='Target class index for targeted attack')
    parser.add_argument('--border_width', type=int, default=4, help='Width of the adversarial border')
    parser.add_argument('--max_iter', type=int, default=50, help='Maximum optimization iterations')
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.attack_type == 'targeted' and args.target_class is None:
        raise ValueError("Target class must be specified for targeted attack")
        
    # Initialize models
    target_model = TargetModel(device)
    fidelity_model = Vgg19Fidelity(device)
    
    attacker = StealthyBorderAttack(
        target_model, 
        fidelity_model, 
        device,
        border_width=args.border_width,
        max_iter=args.max_iter,
        lambda_a=1.0 if args.attack_type == 'targeted' else 1.5, # Paper suggests slightly higher for untargeted
        lambda_f=1000.0
    )
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    image_paths = glob.glob(os.path.join(args.data_dir, '*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print("No images found in data directory.")
        return

    success_count = 0
    total_images = 0
    
    print(f"Found {len(image_paths)} images. Starting attack...")
    
    for img_path in tqdm(image_paths):
        filename = os.path.basename(img_path)
        
        try:
            image = load_image(img_path, device)
            
            # Get Ground Truth (Naive approach: assume model is correct on clean image)
            # In a real evaluation, you should load labels from a file.
            with torch.no_grad():
                init_logits = target_model(image)
                ground_truth = torch.argmax(init_logits, dim=1).item()
            
            # Skip if model is already wrong (standard practice for ASR)
            # But strictly, untargeted attack counts if pred != GT.
            # If it's already wrong, attack is trivial? 
            # Usually we only attack correctly classified images.
            # For this script, we'll attack everything but note this caveat.
            
            target = args.target_class if args.attack_type == 'targeted' else ground_truth
            
            adv_image, is_success = attacker.run(image, args.attack_type, target)
            
            if is_success:
                success_count += 1
            
            total_images += 1
            
            # Save result
            save_path = os.path.join(args.output_dir, f"adv_{filename}")
            save_image(adv_image, save_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if total_images > 0:
        asr = (success_count / total_images) * 100.0
        print("\n" + "="*30)
        print(f"Attack Type: {args.attack_type}")
        print(f"Total Images: {total_images}")
        print(f"Successful Attacks: {success_count}")
        print(f"Attack Success Rate (ASR): {asr:.2f}%")
        print("="*30)
    else:
        print("No images processed successfully.")

if __name__ == "__main__":
    main()