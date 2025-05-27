from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import torch

def get_sampler(labels):
    """ 
    Create a sampler to sample the dataset such that the labels are balanced.
        
        Parameters:
            labels (list[str]): a list of labels assigned to each image in the dataset

        Returns: 
            WeightedRandomSampler 
    """
    # Count how many of each class
    class_counts = np.bincount(labels) # count the number of times each label occurs
    # convert the counts to weights (the label with a lower count will get a higher weight so it will be oversampled and the label with a higher count will get a lower weight so it will be undersampled)
    class_weights = 1.0/class_counts 
    # weight for each image in the dataset (the probability that an image will be selected)
    sample_weights = [class_weights[label] for label in labels]
    #  create sampler. Set replacement to True to allow oversampling (otherwise, the first batches will be balanced but later batches will be unbalanced again as we cannot redraw the same image twice)
    sampler = WeightedRandomSampler(weights=sample_weights,num_samples=len(sample_weights),replacement=True)

    return sampler


def get_dataloader(data_dir, batch_size = 32, num_workers = 2,transform  = None, balance = False):
    """
    Create a dataloader to load the images in the specified directory. It assumes images are arranged in folders with the class's name.
    For example:
        root/Pneumonia
        root/Normal 

        Parameters:
            data_dir (str or pathlib.Path): the root directory path containing the images
            batch_size (int): how many samples per batch to load. Default: 32
            num_workers (int): how many subprocesses to use for data loading. 0 means data will be loaded in the main process. 
            transform (callable): a function/transform that takes in a PIL image or torch.Tensor and returns a transformed version. Default: None
            balance (bool): whether to balance the dataset by performing resampling. Default: False (No)

        Returns:
            A Pytorch DataLoader 
    """

    if transform is None:
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) 
    
    # Load datasets
    ds = datasets.ImageFolder(root=data_dir, transform=transform)
    # Create DataLoaders
    if balance:
        # balance the dataset by weighted sampling
        labels = [label for _, label in ds]
        sampler = get_sampler(np.array(labels))
        dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,sampler=sampler)
    else:
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return dataloader

