import torch
from torchvision.transforms import v2

def _base_transform(img_size=224):
    """ 
    Common preprocessing shared by all transforms

        Returns:
        list[Callable]: List of image transforms
    """
    return [v2.Resize((img_size,img_size)),
            v2.ToImage(),    # converts PIL to v2 tensor image
            v2.Grayscale(num_output_channels=1), # force grayscale
            v2.ToDtype(torch.float32, scale=True), # scales pixel values to [0,1]
            v2.Normalize(mean=[0.5],std=[0.5]) ]# normalize grayscale]


def get_train_transforms(img_size = 224, augment = False):
    """ 
    Creates an returns a composed image transform for the training pipeline

        Parameters:
            img_size (int): output size (n x n) for the resize transformation. Default: 224
            augment (bool): apply common Chest X-ray augmentations (RandomHorizontalFlip, RandomRotation, RandomResizedRecrop, ColorJitter): yes (True) or no (False). If false, just perform base processing (resize, grayscale, normalize). Default: False
    
        Returns: 
            torchvision.transforms.v2.Compose: a composition of image transformations
    """
    aug = []
    if augment:
        aug = [v2.RandomHorizontalFlip(p=0.5), # randomly flip the image with 50% probability, variations in patient position
               v2.RandomRotation(degrees=10), # randomly rotate the image in the range -10 to 10, mimic slight positioning differences during scanning
               v2.RandomResizedCrop(size=img_size,scale=(0.9,1.0)), #randomly crop the image and resize it,variability in cropping or zooming level
               v2.ColorJitter(brightness=0.1, contrast=0.1) # variability in image acquisition (exposure, contrast) 
               ]  
    
    return v2.Compose(aug + _base_transform(img_size))
        

def get_val_transforms(img_size = 224):
    """
    Creates and returns a composed image transform for validation. No random augmentation, just apply resizing, convert to grayscale, and normalization
        Parameters:
            img_size (int): output size (n x n) for the resize transformation. Default: 224

        Returns: 
            torchvision.transforms.v2.Compose: a composition of image transformations
    """
    return v2.Compose(_base_transform(img_size))
    
