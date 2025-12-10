import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ImageNet statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_preprocessing_transforms():
    """
    Returns the transformation pipeline described in the paper.
    1. Resize to 224x224
    2. Convert to Tensor (scales to [0, 1])
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def normalize(tensor):
    """
    Applies ImageNet normalization to a [0, 1] tensor.
    Input: (B, C, H, W)
    """
    mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(STD).view(1, 3, 1, 1).to(tensor.device)
    return (tensor - mean) / std

def denormalize(tensor):
    """
    Reverses ImageNet normalization to get back to [0, 1].
    """
    mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(STD).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

def load_image(path, device):
    """Loads an image, preprocesses it, and puts it on device."""
    image = Image.open(path).convert('RGB')
    transform = get_preprocessing_transforms()
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dim
    return image_tensor

def save_image(tensor, path):
    """Saves a [0, 1] tensor as an image."""
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = torch.clamp(tensor, 0, 1)
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    img.save(path)

def extract_borders(image_tensor, border_width):
    """
    Extracts the 4 borders of the image.
    image_tensor: (B, C, H, W)
    Returns: A list of 4 tensors [Top, Bottom, Left, Right]
    """
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    
    # Top border: (B, C, bw, W)
    top = image_tensor[:, :, :border_width, :]
    # Bottom border: (B, C, bw, W)
    bottom = image_tensor[:, :, h-border_width:, :]
    # Left border (excluding top/bottom corners to avoid overlap if strict, 
    # but paper implies simple rectangular masks. We take full height strips 
    # or side strips. Let's take side strips corresponding to the 'middle' 
    # to avoid double counting corners, or just full strips. 
    # Paper Fig 3 implies corners are handled. Let's stick to full height for simplicity 
    # but strictly we should treat corners carefully. 
    # Simple strategy: Extract full width top/bottom, and height-2*bw left/right.
    
    left = image_tensor[:, :, border_width:h-border_width, :border_width]
    right = image_tensor[:, :, border_width:h-border_width, w-border_width:]
    
    return top, bottom, left, right

def stack_borders(image_tensor, border_width):
    """
    Extracts borders and stacks them vertically to create a 'border image'
    for the fidelity model, as per Section IV.B "Stacking Borders".
    """
    top, bottom, left, right = extract_borders(image_tensor, border_width)
    
    # We need to reshape left/right to stack them vertically with top/bottom.
    # Current shapes:
    # Top/Bottom: (B, C, bw, W) -> e.g. (1, 3, 4, 224)
    # Left/Right: (B, C, H-2bw, bw) -> e.g. (1, 3, 216, 4)
    
    # The paper says "The borders are repeated to satisfy the minimum size required".
    # VGG needs decent spatial dims. 
    # Let's rotate the side borders to match width or just concatenate raw pixels?
    # Style transfer usually works on texture.
    # Let's permute Left/Right to be (B, C, bw, H-2bw) so they are "horizontal" strips
    # and then concat everything vertically.
    
    left_permuted = left.permute(0, 1, 3, 2)   # (B, C, 4, 216)
    right_permuted = right.permute(0, 1, 3, 2) # (B, C, 4, 216)
    
    # Resize side strips to width 224 to match top/bottom for clean stacking?
    # Or just stack them all as long strips. 
    # To enable easy batch processing, let's just resize the width of sides to W 
    # or keep them as is and pad?
    # Simplest approach for style transfer: Concat everything into one long strip.
    # But tensor concatenation requires matching dimensions.
    # Let's resize left/right to length W (224).
    
    target_width = image_tensor.shape[3]
    
    # Helper to resize length
    def resize_strip(strip):
        return torch.nn.functional.interpolate(strip, size=(border_width, target_width), mode='bilinear')

    # Note: permuting effectively rotates the image content 90 deg. 
    # This might affect "content" representation but "style" (texture) is usually rotation invariant-ish.
    # However, for "alignment", we compare Clean Border vs Adversarial Border.
    # As long as we do the same operation to both, the loss is valid.
    
    stacked = torch.cat([
        top, 
        bottom, 
        resize_strip(left_permuted), 
        resize_strip(right_permuted)
    ], dim=2) # Concat along height
    
    # We repeat the block to ensure it's large enough for VGG (optional, but paper mentions it)
    # If total height is small (4*4 = 16px), VGG downsampling will kill it.
    # We need to repeat it to get at least ~32-64px height.
    
    while stacked.shape[2] < 64:
        stacked = torch.cat([stacked, stacked], dim=2)
        
    return stacked

def create_border_mask(h, w, border_width, device):
    """
    Creates a binary mask where 1 = border (attack region), 0 = inner.
    """
    mask = torch.zeros((1, 3, h, w), device=device)
    
    # Fill borders with 1
    mask[:, :, :border_width, :] = 1 # Top
    mask[:, :, h-border_width:, :] = 1 # Bottom
    mask[:, :, :, :border_width] = 1 # Left
    mask[:, :, :, w-border_width:] = 1 # Right
    
    return mask

def truncation_loss(adv_image):
    """
    Eq 6: L_T = sum | int(pixel * 255)/255.0 - pixel |
    Minimizes the rounding error when saving as integer image.
    """
    scaled = adv_image * 255.0
    target_int = torch.round(scaled).detach() # The closest integer
    # Loss is L1 distance between current float value and the closest integer
    return torch.abs(target_int/255.0 - adv_image).mean()