import os
import torch
from torch import nn
from torchvision import models



def get_resnet18(pretrained=True, grayscale=True, freeze = False):
    """
    Loads a ResNet-18 model for binary classification.
    
        Parameters:
            pretrained (bool): load pretrained weights if True. Default: True.
            grayscale (bool): Modify first layer to accept grayscale input. Default: True.
            freeze (bool): freeze all layers except the final classification head (reduces overfitting), and, if using grayscale images, the first conv1 layer
        Returns:
            model (nn.Module): ResNet-18 model"""
    # Load pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    if grayscale:
        # modify the input layer to accept 1-channel input
        # Save the pretrained weights from the original 3-channel conv1
        pretrained_conv1_weights = model.conv1.weight.clone()

        # replace input layer to accept 1-channel input
        model.conv1 = nn.Conv2d(1,64,kernel_size=7, stride=2,padding=3,bias=False)

        # Copy weights from original input layer by averaging accross channels
        # This helps retain pretrained knowledge even though the input layer has changed
        with torch.no_grad():
            model.conv1.weight[:] = pretrained_conv1_weights.mean(dim=1, keepdim=True)

    # adapt for binary classification
    num_features = model.fc.in_features
    # replace the classifier head.
    model.fc = nn.Linear(num_features,1)

    if freeze:
        # Freeze all layers except input layer (if grayscale) and classifier head
        for name,param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            elif ("conv1" in name) and grayscale:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return model

def get_resnet34(pretrained=True, grayscale=True, freeze = False):
    """
    Loads a ResNet-34 model for binary classification.
        Parameters:
            pretrained (bool): load pretrained weights if True. Default: True.
            grayscale (bool): Modify first layer to accept grayscale input. Default: True
            freeze (bool): freeze all layers except the final classification head (reduces overfitting), and, if using grayscale images, the first conv1 layer
        Returns:
            model (nn.Module): ResNet-34 model"""
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

    if grayscale:
        # modify the input layer to accept 1-channel input
        # Save the pretrained weights from the original 3-channel conv1
        pretrained_conv1_weights = model.conv1.weight.clone()

        # replace input layer to accept 1-channel input
        model.conv1 = nn.Conv2d(1,64,kernel_size=7, stride=2,padding=3,bias=False)

        # Copy weights from original input layer by averaging accross channels
        # This helps retain pretrained knowledge even though the input layer has changed
        with torch.no_grad():
            model.conv1.weight[:] = pretrained_conv1_weights.mean(dim=1, keepdim=True)

    # adapt for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features,1)

    if freeze:
        # Freeze all layers except input layer (if grayscale) and classifier head
        for name,param in model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            elif ("conv1" in name) and grayscale:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return model

