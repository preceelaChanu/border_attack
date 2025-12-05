import torch
import torch.nn as nn
import torch.optim as optim
from utils import create_border_mask, stack_borders, truncation_loss

class StealthyBorderAttack:
    def __init__(self, target_model, fidelity_model, device, 
                 border_width=4, lambda_a=1.0, lambda_f=1000.0, 
                 max_iter=100):
        self.target_model = target_model
        self.fidelity_model = fidelity_model
        self.device = device
        self.border_width = border_width
        self.lambda_a = lambda_a
        self.lambda_f = lambda_f
        self.max_iter = max_iter
        
        self.mask = None # Created dynamically based on image size

    def _targeted_loss(self, logits, target_class):
        # Eq 4: -log(p(x', T))
        # CrossEntropyLoss computes -log(softmax(logits)[target])
        criterion = nn.CrossEntropyLoss()
        target_tensor = torch.tensor([target_class], device=self.device)
        return criterion(logits, target_tensor)

    def _untargeted_loss(self, logits, original_class):
        # Eq 3: L_margin = Z(x')_y - max(Z(x')_i) for i != y
        # We want to minimize L_A. 
        # If we minimize (Z_true - Z_max_other), we make Z_true smaller and Z_max_other larger.
        
        probs = torch.softmax(logits, dim=1)
        z_y = logits[0, original_class]
        
        # Get max of other classes
        logits_others = logits.clone()
        logits_others[0, original_class] = -float('inf')
        z_max_other, _ = torch.max(logits_others, dim=1)
        
        l_margin = z_y - z_max_other
        
        # Eq 4: L_A = max(L_margin, 0.1 * L_margin) (Leaky ReLU logic)
        # Note: If l_margin is positive (classified correctly), we want to reduce it.
        # If l_margin is negative (misclassified), we still push it (0.1 slope).
        l_a = torch.max(l_margin, 0.1 * l_margin)
        return l_a

    def run(self, image, attack_type='targeted', target=None):
        """
        image: (1, 3, H, W) tensor, normalized [0, 1]
        attack_type: 'targeted' or 'untargeted'
        target: target class index (for targeted) or original class index (for untargeted)
        """
        b, c, h, w = image.shape
        self.mask = create_border_mask(h, w, self.border_width, self.device)
        inverse_mask = 1 - self.mask
        
        # Initialize adversarial border: can start from original border
        # We optimize the whole image but mask gradients/updates
        adv_image = image.clone().detach().requires_grad_(True)
        
        # We use LBFGS as suggested in standard style transfer and optimization attacks
        optimizer = optim.LBFGS([adv_image], max_iter=20, history_size=10) # max_iter per step
        
        clean_stacked_borders = stack_borders(image.detach(), self.border_width)
        
        iteration = 0
        success = False
        
        print(f"Starting {attack_type} attack...")

        while iteration < self.max_iter:
            def closure():
                optimizer.zero_grad()
                
                # Enforce pixel constraints [0, 1]
                with torch.no_grad():
                    adv_image.clamp_(0, 1)
                    # Reset inner part to original to ensure we ONLY modify borders
                    # Although mask handles gradients, LBFGS might step out.
                    adv_image.data = adv_image.data * self.mask + image.data * inverse_mask
                
                logits = self.target_model(adv_image)
                
                # Attack Loss
                if attack_type == 'targeted':
                    l_attack = self._targeted_loss(logits, target)
                else:
                    l_attack = self._untargeted_loss(logits, target) # target here is ground truth
                
                # Fidelity Loss
                adv_stacked = stack_borders(adv_image, self.border_width)
                l_c, l_s = self.fidelity_model.compute_loss(adv_stacked, clean_stacked_borders)
                l_fidelity = l_c + l_s # Weighted inside model or here? Paper: sum(w*Lc + w*Ls)
                
                # Truncation Loss
                l_trunc = truncation_loss(adv_image * self.mask) # Only borders
                
                # Total Loss
                total_loss = (self.lambda_a * l_attack) + (self.lambda_f * l_fidelity) + l_trunc
                
                if adv_image.grad is not None:
                     # Zero out gradients for inner region before backward (safety)
                     adv_image.grad.data.mul_(self.mask)

                total_loss.backward()
                
                # Mask gradients again after backward
                adv_image.grad.data.mul_(self.mask)
                
                return total_loss

            optimizer.step(closure)
            
            # Check success
            with torch.no_grad():
                logits = self.target_model(adv_image)
                pred = torch.argmax(logits, dim=1).item()
                
                if attack_type == 'targeted':
                    if pred == target:
                        success = True
                        break
                else:
                    if pred != target: # target is ground truth
                        success = True
                        break
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"Iter {iteration}: Prediction {pred}")

        # Final cleanup
        with torch.no_grad():
            adv_image.clamp_(0, 1)
            adv_image.data = adv_image.data * self.mask + image.data * inverse_mask
            
        return adv_image, success