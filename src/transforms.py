import torch
from torchvision.transforms import v2

def _base_transform(img_size=224, to_rgb=False):
    """ 
    Common preprocessing shared by all transforms

        Parameters:
            img_size (int): size of the output image
            to_rgb (bool): If True, converts grayscale to 3 channels. Default: False
        Returns:
            list[Callable]: List of image transforms
    """

    # Imagenet stats (for 3-channel input)
    # ImageNet stats (for 3-channel input)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Gray  stats (for 1 channel input)
    gray_mean = [0.5]
    gray_std = [0.5]

    # color conversion
    if to_rgb:
        color_transform = v2.Grayscale(num_output_channels=3) # copy to  3 channels
        normalize = v2.Normalize(mean=imagenet_mean, std=imagenet_std) # apply stats of imagenet (as the models we use to_rgb with are trained on imagenet)
    else:
        color_transform = v2.Grayscale(num_output_channels=1)
        normalize = v2.Normalize(mean=gray_mean, std=gray_std)

    return [color_transform, # force grayscale
            v2.Resize((img_size,img_size)),
            v2.ToDtype(torch.float32, scale=True), # scales pixel values to [0,1]
            normalize ]# normalize grayscale]


def get_transforms(img_size = 224, to_rgb=False, augment=False):
    """ 
    Creates an returns a composed image transform 

        Parameters:
            img_size (int): output size (n x n) for the resize transformation. Default: 224
            to_rgb (bool): If True, convert 1-channel grayscale to 3 channel. Default: False
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
    
    return v2.Compose([v2.ToImage()] + aug + _base_transform(img_size, to_rgb))
        