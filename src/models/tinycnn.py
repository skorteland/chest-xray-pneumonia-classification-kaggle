import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

def conv_block(input_channels, output_channels, use_batchnorm=False):
    """
    Create convolution block 
        Parameters:
            input_channels (int): number of input channels into the Conv block
            output_channels (int): number of output channels (number of filters) 
            use_batchnorm (bool): use a BatchNormalization layer yes (True) or no (False). Default: False
        Returns: 
            torch.nn.Sequential:  A sequential container of layers forming the block, containing Conv2D, BatchNorm2D (optional), ReLU and MaxPool2D layers. 
    """
    layers = [nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,padding=1)]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(output_channels))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # MaxPool after ReLU as in this case it is equivalent, but more efficient that ReLU after MaxPool
    return nn.Sequential(*layers)


class TinyCNN(nn.Module):
    def __init__(self,img_size,in_channels=1,use_batchnorm=True,use_dropout=True, dropout_prob=0.3):
        """ 
        Building blocks of the simple convolutional neural network.
        
        Parameters:
        - img_size (int): size of the images used for training and testing (the images should all have the same size).
        - in_channels (int): number of channels in the input image (for grayscale images this is 1). Default: 1.
        - use_batchnorm (bool): whether to use batch normalization. Default: True
        - use_dropout (bool): whether to use dropout layers. Default: True
        - dropout_prob (float): if use_dropout=True, then this is the dropout rate used in the dropout layer. Default: 0.3
        """

        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(in_channels, 16, use_batchnorm),
            conv_block(16,32,use_batchnorm)
        )

        decoder_layers = [nn.Flatten()]
        if use_dropout:
            decoder_layers.append(nn.Dropout(dropout_prob))
        
        # assuming maxpool after every conv layer, the input size for the first fully connected layer should be:
        # 16*img_size/4 * img_size/4
        input_size = int(32 * img_size/4 * img_size/4)

        # Fully connected layers
        decoder_layers.append(nn.Linear(input_size,64)) # fc1
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(64,1)) # fc2, binary classification

        self.decoder = nn.Sequential(*decoder_layers)

        
    def forward(self, x):
        """
        Define the forward pass of the neural network.
        
        Parameters:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: the output tensor after passing through the network.
        """
        x = self.encoder(x)
        #x = torch.flatten(x,start_dim=1) # flatten the tensor
        x = self.decoder(x)
        #Note: Apply fully connected layer. do not apply sigmoid as the last step as we use the BCEWithLogitsLoss function which already has sigmoid
        
        return x

def get_tinycnn(img_size, in_channels,use_batchnorm=True,use_dropout=True,dropout_prob=0.3):
    return TinyCNN(img_size=img_size,in_channels=in_channels,use_batchnorm=use_batchnorm,use_dropout=use_dropout,dropout_prob=dropout_prob)
