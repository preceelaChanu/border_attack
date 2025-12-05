import torch
import torch.nn as nn
import torchvision.models as models
from utils import normalize

class TargetModel(nn.Module):
    def __init__(self, device):
        super(TargetModel, self).__init__()
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True).to(device)
        self.model.eval()
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device

    def forward(self, x):
        # x is [0, 1]. Normalize before passing to ResNet
        x_norm = normalize(x)
        return self.model(x_norm)

class Vgg19Fidelity(nn.Module):
    """
    VGG19 feature extractor for Content and Style loss.
    Extracts features from specific layers.
    """
    def __init__(self, device):
        super(Vgg19Fidelity, self).__init__()
        vgg = models.vgg19(pretrained=True).features.to(device)
        self.model = vgg.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Layers often used for Style/Content in style transfer
        # Paper doesn't specify exact layers, using standard Gatys et al. config
        self.content_layers = ['conv_4'] 
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        self.layer_map = {
            '0': 'conv_1',
            '5': 'conv_2',
            '10': 'conv_3',
            '19': 'conv_4',
            '28': 'conv_5',
        }

    def forward(self, x):
        # x should be the stacked borders, normalized
        x = normalize(x)
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layer_map:
                features[self.layer_map[name]] = x
        return features

    def compute_loss(self, adv_stacked, clean_stacked):
        adv_feats = self(adv_stacked)
        clean_feats = self(clean_stacked)
        
        loss_c = 0.0
        loss_s = 0.0
        
        # Content Loss
        for layer in self.content_layers:
            loss_c += torch.mean((adv_feats[layer] - clean_feats[layer]) ** 2)
            
        # Style Loss (Gram Matrix)
        for layer in self.style_layers:
            a = adv_feats[layer]
            c = clean_feats[layer]
            
            b, ch, h, w = a.shape
            
            a = a.view(b, ch, h * w)
            c = c.view(b, ch, h * w)
            
            gram_a = torch.bmm(a, a.transpose(1, 2)) / (ch * h * w)
            gram_c = torch.bmm(c, c.transpose(1, 2)) / (ch * h * w)
            
            loss_s += torch.mean((gram_a - gram_c) ** 2)
            
        return loss_c, loss_s