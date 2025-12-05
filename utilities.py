import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ImageNet statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_preprocessing_transforms():
    """
    Returns the transformation pipeline.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def normalize(tensor):
    """
    Applies ImageNet normalization. 
    Expects tensor in range [0, 1].
    """
    mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(STD).view(1, 3, 1, 1).to(tensor.device)
    return (tensor - mean) / std

def load_tensor_data(path, device):
    """
    Loads a pre-processed .pt tensor file.
    """
    tensor = torch.load(path, map_location=device)
    return tensor

def save_image(tensor, path):
    """Saves a [0, 1] tensor as an image."""
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = torch.clamp(tensor, 0, 1)
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    img.save(path)

def extract_borders(image_tensor, border_width):
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    top = image_tensor[:, :, :border_width, :]
    bottom = image_tensor[:, :, h-border_width:, :]
    # Extract vertical strips for left/right
    left = image_tensor[:, :, border_width:h-border_width, :border_width]
    right = image_tensor[:, :, border_width:h-border_width, w-border_width:]
    return top, bottom, left, right

def stack_borders(image_tensor, border_width):
    """
    Extracts borders and stacks them for the fidelity model.
    """
    top, bottom, left, right = extract_borders(image_tensor, border_width)
    
    # Permute side borders to be horizontal strips for stacking
    left_permuted = left.permute(0, 1, 3, 2)   
    right_permuted = right.permute(0, 1, 3, 2) 
    
    target_width = image_tensor.shape[3]
    
    def resize_strip(strip):
        return torch.nn.functional.interpolate(strip, size=(border_width, target_width), mode='bilinear', align_corners=False)

    stacked = torch.cat([
        top, 
        bottom, 
        resize_strip(left_permuted), 
        resize_strip(right_permuted)
    ], dim=2) 
    
    # Repeat to ensure sufficient height for VGG
    while stacked.shape[2] < 64:
        stacked = torch.cat([stacked, stacked], dim=2)
        
    return stacked

def create_border_mask(h, w, border_width, device):
    mask = torch.zeros((1, 3, h, w), device=device)
    mask[:, :, :border_width, :] = 1 
    mask[:, :, h-border_width:, :] = 1 
    mask[:, :, :, :border_width] = 1 
    mask[:, :, :, w-border_width:] = 1 
    return mask

def truncation_loss(adv_image):
    scaled = adv_image * 255.0
    target_int = torch.round(scaled).detach()
    return torch.abs(target_int/255.0 - adv_image).mean()