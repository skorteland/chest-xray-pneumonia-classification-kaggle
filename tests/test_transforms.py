import pytest
import torch
import numpy as np
from PIL import Image
from src.transforms import get_transforms

# helper function: create a dummy grayscale image as a PIL image
def create_dummy_image(size=(256,256), constant_value=128): 
    arr = np.ones(size, dtype=np.uint8) * constant_value
    return Image.fromarray(arr, mode='L')  # Grayscale
    
@pytest.mark.parametrize("augment",[False,True])
def test_transforms_output_shape_and_size(augment):
    img_size = 224
    image =create_dummy_image()
    transform = get_transforms(img_size=img_size, to_rgb=False, augment=augment)

    transformed_image = transform(image)

    assert isinstance(transformed_image,torch.Tensor)
    assert transformed_image.shape == (1,img_size,img_size),f"Expected (1,{img_size},{img_size}), got {transformed_image.shape}"
    assert transformed_image.dtype == torch.float32


@pytest.mark.parametrize("augment",[False,True])
def test_transforms_value_range(augment):
    img_size = 224
    image =create_dummy_image()
    transform = get_transforms(img_size=img_size, to_rgb=False, augment=augment)

    image =create_dummy_image()
    transformed_image = transform(image)

    # check values are in expected normalized range
    assert transformed_image.min() >= -3.0 and transformed_image.max() <= 3.0, f"Pixel values out of expected range: min={transformed_image.min()}, max={transformed_image.max()}"

def test_transforms_grayscale():
    img_size = 224
    
    transform = get_transforms(img_size=224, to_rgb=False, augment=False)
    image =create_dummy_image()
    transformed_image = transform(image)

    # check that grayscale has 1 channel
    assert transformed_image.shape[0] == 1, f"Expected 1 channel, got {transformed_image.shape[0]}"
    
def test_transforms_rgb():
    img_size = 224
    transform = get_transforms(img_size=img_size, to_rgb=True, augment=False)
    image =create_dummy_image()
    transformed_image = transform(image)

    # RGB should have 3 channels
    assert transformed_image.shape[0] ==3, f"Expected 3 channels, got {transformed_image.shape[0]} channels"

def test_transforms_augmentation_changes_image():
    image = create_dummy_image()

    # Augmented and base transforms
    torch.manual_seed(42)
    aug_transform = get_transforms(img_size=224, to_rgb=False, augment=True)
    torch.manual_seed(42)
    base_transform = get_transforms(img_size=224, to_rgb=False, augment=False)

    image_aug = aug_transform(image)
    image_base = base_transform(image)

    assert not torch.allclose(image_aug, image_base), "Augmentation should alter the image"

def test_transforms_color_jitter_affects_intensity():
    image = create_dummy_image(constant_value=100)
    transform = get_transforms(img_size=224, to_rgb=False, augment=True)

    torch.manual_seed(0)

    image_aug = transform(image)
    intensity = image_aug.mean().item()

    # For grayscale image, normalized [0,1] -> mean 0.5. Jitter should move this
    assert intensity < 0.49 or intensity > 0.51, f"Expected brightness jitter. Got mean {intensity:.3f}"

def test_random_augmentation_changes_pixel_locations():
    image = create_dummy_image()
    img_size = 224
    transform = get_transforms(img_size=img_size, to_rgb=False, augment=True)

    torch.manual_seed(1)
    img_aug1 = transform(image)

    torch.manual_seed(1)
    img_aug2 = transform(image)

    # using the same seed, we should get a deterministic result
    assert torch.allclose(img_aug1,img_aug2), "Augmentation should be deterministic with fixed seed"

    torch.manual_seed(2)
    img_aug3 = transform(image)

    # Using different seed, we should get different result
    assert not torch.allclose(img_aug1, img_aug3), "Augmentations should be different with different seeds"

def test_transforms_resized_crop_keeps_size():
    img_size = 224
    image = create_dummy_image()
    transform = get_transforms(img_size=224, to_rgb=False, augment=True)

    image_aug = transform(image)

    assert image_aug.shape[-2:] == (img_size, img_size), "Image after crop/resize must match target size"