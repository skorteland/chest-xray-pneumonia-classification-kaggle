import os
import torch
from torch import nn
from torchvision import models

def get_mobilenet(pretrained=True, grayscale=True, freeze=False):
    """
    Loads a MobileNetV2 model for binary classification.
    
        Paramaters:
            pretrained (bool): load pretrained weights if True. Default: True.
            grayscale (bool): Modify first layer to accept grayscale input. Default: True.
            freeze (bool): freeze all layers except the final classification head (reduces overfitting)
        Returns:
            model (nn.Module): MobileNetV2 model
    """

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)

    # Modify first conv layer to accept grascale (1 channel) input
    if grayscale:
        model.features[0][0] = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False)

    # Replace classifier head for binary classification
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel,1))

    if freeze:
        # freeze backbone except first conv and classifier
        for name,param in model.named_parameters():
            param.requires_grad =False

        # Unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Unfreeze input layer (if grayscale)
        if grayscale:
            for param in model.features[0][0].parameters():
                param.requires_grad = True
       

    return model

    