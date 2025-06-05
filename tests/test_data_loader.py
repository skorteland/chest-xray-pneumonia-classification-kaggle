import os
import sys
from pathlib import Path
import pytest
import tempfile
import shutil
import torch
import numpy as np
from collections import Counter
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler, RandomSampler
from torchvision.transforms import v2
from unittest.mock import patch, MagicMock
module_path = os.path.abspath(os.path.join('..')) # project root
# add project root to sys.path
if module_path not in sys.path:
    sys.path.append(module_path) 
from src.data_loader import get_dataloader, get_sampler
from src.transforms import get_train_transforms

THIS_DIR = Path(__file__).parent

def create_fake_dataset(root_dir):
    # helper to create fake images in folder/class structure
    classes = ['NORMAL','PNEUMONIA']
    for class_name in classes:
        class_dir = Path(root_dir) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5): # 5 images per class
            img = Image.new("RGB", (64,64), color = (i*10,i*10,i*10))
            img.save(os.path.join(class_dir,f'{class_name}_{i}.png'))


# pytest fixture provides setup code that can be injected into tests by passing it as an argument. 
@pytest.fixture
def fake_data_dir():
    tmp_dir = tempfile.mkdtemp()
    create_fake_dataset(tmp_dir)
    yield tmp_dir # return the filepath from the function and continue
    shutil.rmtree(tmp_dir) # cleanup

def test_get_sampler_returns_balanced_sampler():
    labels = np.array([0] * 90 + [1] * 10) # create unbalanced labels
    sampler = get_sampler(labels)

    assert isinstance(sampler, WeightedRandomSampler)
    assert len(sampler) == len(labels)

    # Draw many samples to estimate class balance
    sampled_indices = list(sampler)[:1000]
    sampled_labels = [labels[i] for i in sampled_indices]
    class_counts = Counter(sampled_labels)

    # expect roughly equal sampling (within 10%)
    total = sum(class_counts.values())
    proportions = np.array(list(class_counts.values())) / total
    assert np.allclose(proportions[0], proportions[1],rtol=0.1),f"Unbalanced: {class_counts}"

# pytest sees that fake_data_dir is annotated as a fixture, and thus automatically runs the fixture before the test. It then injects the returned value into the test function.
def test_get_dataloader_batch_size(fake_data_dir):
    batch_size = 16
    dataloader = get_dataloader(data_dir=fake_data_dir, batch_size=batch_size)

    images,labels = next(iter(dataloader))

    assert images.shape[0] <= batch_size # last batch may be smaller
    assert labels.shape[0] == images.shape[0]

# pytest sees that fake_data_dir is annotated as a fixture, and thus automatically runs the fixture before the test. It then injects the returned value into the test function.
def test_get_dataloader_default(fake_data_dir):
    dataloader = get_dataloader(data_dir=fake_data_dir)
    
    assert isinstance(dataloader,DataLoader)
    batch = next(iter(dataloader))
    images,labels = batch
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)

    assert images.shape[0] <= 32 # default batch size
    assert images.ndim == 4 # batch, channels, height,width
    # default transform has been applied
    assert isinstance(images, torch.Tensor)
    assert images.dtype == torch.float32 
    assert images.max() <= 1.0 and images.min() >= 0.0 # values are normalized between 0 and 1


def test_get_dataloader_with_custom_transform(fake_data_dir):
    # create a transform
    img_size = 32
    
    custom_transform = v2.Compose([
            v2.Resize((img_size,img_size)),
            v2.ToImage(),    # converts PIL to v2 tensor image
            v2.Grayscale(num_output_channels=1), # force grayscale
            v2.ToDtype(torch.float32, scale=True) # scales pixel values to [0,1]
    ])

    dataloader = get_dataloader(data_dir=fake_data_dir, transform = custom_transform)
    images, _ = next(iter(dataloader))
    
    assert images.shape[-2:] == (img_size, img_size) # check resize
    assert images.shape[-3] == 1 # check grayscale (1 channel)
    
    

@patch("src.data_loader.get_sampler")
def test_get_dataloader_uses_sampler(mock_get_sampler,fake_data_dir):
    mock_sampler = MagicMock()
    mock_get_sampler.return_value = mock_sampler

    dataloader = get_dataloader(data_dir=fake_data_dir, balance=True)

    # assert get_sampler was called
    assert mock_get_sampler.called, "get_sampler should be called when balance=True"

    # assert returned DataLoader is using the mocked sampler
    assert dataloader.sampler == mock_sampler, "Sampler was not properly used in DataLoader"

def test_get_dataloader_without_balance(fake_data_dir):
    dataloader = get_dataloader(data_dir=fake_data_dir, balance=False)

    # DataLoader should use default sampler (shuffle =True -> RandomSampler)
    assert isinstance(dataloader.sampler, RandomSampler)