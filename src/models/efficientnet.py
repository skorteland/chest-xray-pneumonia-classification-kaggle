import os
import torch
from torch import nn
from torchvision import models

def get_efficientnetB0(pretrained=True, freeze=False):
    """
    Loads an EfficientNet-B0 model for binary classification. Expects 3-channel input (RGB or grayscale repeated to RGB)
    
        Parameters:
            pretrained (bool): load pretrained weights (on ImageNet) if True. Default: True.
            freeze (bool): freeze all layers except the final classification head (reduces overfitting)
    
        Returns:
            model (torch.nn.Module): EfficientNet-B0s model with binary classifier head
    """
    # Load EfficientNet B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)


    # Replace classifier head for binary classification
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features,1))

    if freeze:
        # freeze the backbone layers (everything except classifier)
        for param in model.features.parameters():
            param.requires_grad = False

    return model
     
    