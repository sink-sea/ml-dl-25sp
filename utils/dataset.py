# Load Datasets

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

def get_cifar10_dataloader(batch_size=64, num_workers=2, train=True):
    """
    Returns a DataLoader for the CIFAR-10 dataset.
    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        train (bool): If True, returns the training set; otherwise, returns the test set
    Returns:
        DataLoader: DataLoader for the CIFAR-10 dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_ImageNet_dataloader(batch_size=64, num_workers=2, train=True):
    """
    Returns a DataLoader for the ImageNet dataset.
    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        train (bool): If True, returns the training set; otherwise, returns the test set
    Returns:
        DataLoader: DataLoader for the ImageNet dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # load from ./data/tiny-imagenet-200
    dataset = torchvision.datasets.ImageFolder(
        root='./data/tiny-imagenet-200/train' if train else './data/tiny-imagenet-200/val',
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_custom_dataset_dataloader(root_dir, batch_size=64, num_workers=2, train=True):
    """
    Returns a DataLoader for a custom dataset.
    Args:
        root_dir (str): Root directory of the dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        train (bool): If True, returns the training set; otherwise, returns the test set
    Returns:
        DataLoader: DataLoader for the custom dataset.
    """
    class CustomDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = os.path.curdir + '/' + root_dir
            self.root_dir += '/train' if train else '/test'
            print(f"Loading dataset from {self.root_dir}")
            self.transform = transform
            self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.jpg') or fname.endswith('.png')]
            self.labels = [int(fname.split('_')[0]) for fname in os.listdir(root_dir) if fname.endswith('.jpg') or fname.endswith('.png')]

        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CustomDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def get_dataloader(args, train=True):
    """
    Returns a DataLoader based on the dataset specified in the arguments.
    Args:
        args (Namespace): Parsed command line arguments.
        train (bool): If True, returns the training set; otherwise, returns the test set
    Returns:
        DataLoader: DataLoader for the specified dataset.
    """
    if args.dataset == 'cifar10':
        return get_cifar10_dataloader(args.batch_size, args.num_workers, train)
    elif args.dataset == 'imagenet':
        return get_ImageNet_dataloader(args.batch_size, args.num_workers, train)
    else:
        return get_custom_dataset_dataloader(args.dataset, args.batch_size, args.num_workers, train)