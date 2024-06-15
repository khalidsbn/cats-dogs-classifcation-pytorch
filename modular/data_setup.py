"""
Contains functionality for creating PyTorch DataLoader's for
image classification data.
"""
import os
from PIL import Image
from typing import List, Tuple
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

NUM_WORKERS = os.cpu_count()

class CustomDataset(Dataset):
    """
    A custom dataset class for loading and transforming images.

    Args:
        file_list (List[str]): A list of file paths to the images.
        transform (callable, optional): An optional transform to be applied to the images.

    Attributes:
        file_list (List[str]): Stores the list of image file paths.
        transform (callable, optional): Stores the transform to be applied to the images.

    Methods:
        __len__():
            Returns:
                int: The total number of images in the dataset.

        __getitem__(idx):
            Args:
                idx (int): The index of the image to retrieve.

            Returns:
                tuple: A tuple containing:
                    - img (PIL.Image.Image): The transformed image.
                    - label (int): The label of the image, where 'dog' is 1, 'cat' is 0, and any undefined label is -1.
    """
    def __init__(self, file_list: List[str], transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0
        else:
            label = -1 # Undefined label

        return img, label

def create_dataloaders(train_list: List[str],
                       valid_list: List[str],
                       test_list: List[str],
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = NUM_WORKERS) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates training, validation, and testing DataLoaders.

    Args:
      train_list: List of paths to training images.
      valid_list: List of paths to validation images.
      test_list: List of paths to testing images.
      transform: torchvision transforms to perform on data.
      batch_size: Number of samples per batch in each of the DataLoaders.
      num_workers: Number of subprocesses to use for data loading.

    Returns:
      A tuple of (train_dataloader, valid_dataloader, test_dataloader).
    """
    # Create datasets
    train_data = CustomDataset(train_list, transform=transform)
    valid_data = CustomDataset(valid_list, transform=transform)
    test_data = CustomDataset(test_list, transform=transform)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True # for more on pin memory, see PyTorch docs: https://pytorch.org/docs/stable/data.html
    )

    valid_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, valid_dataloader, test_dataloader
