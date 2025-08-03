import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch
from typing import List

def streamingllm_attention_density(
    global_size: int = 4,
    band_size: int = 1024,
    kv_seq_len: int = 8192,
):  
    num_total = 0
    num_attended = 0

    for i in range(kv_seq_len):
        for j in range(kv_seq_len):
            if i < j:
                continue
            num_total += 1

            if (j < global_size) or (i - j < band_size):
                num_attended += 1
    
    return num_attended / num_total

def streamingllm_kv_cache_density(
    global_size: int = 4,
    band_size: int = 1024,
    kv_seq_len: int = 8192,
):
    if global_size + band_size > kv_seq_len:
        return 1.0

    else:
        return (global_size + band_size) / kv_seq_len

### calculate lut density directly using block. This may incur small error since on diagonal, the number of attended elements is not exactly the whole block
def count_distinct_elements_per_row(
    lut: torch.IntTensor,
):
    assert lut.dim() == 2
    M, N = lut.shape
    distinct_counts = torch.zeros(M, dtype=torch.long)

    for i in range(M):
        row_unique = torch.unique(lut[i])
        distinct_counts[i] = row_unique.numel()

    return distinct_counts

def lut_single_layer_attention_density(
    lut: torch.IntTensor,
    block_size: int = 64
):
    assert lut.dim() == 3

    kv_seq_block_len = lut.shape[1]

    num_total = lut.shape[0] * (kv_seq_block_len * (kv_seq_block_len + 1)) // 2
    num_attended = 0

    for i in range(lut.shape[0]):
        num_attended += count_distinct_elements_per_row(lut[i]).sum().item()

    return num_attended / num_total

def lut_single_layer_kv_cache_density(
    lut: torch.IntTensor,
    block_size: int = 64
):
    assert lut.dim() == 3

    kv_seq_block_len = lut.shape[1]

    num_total = lut.shape[0] * kv_seq_block_len
    num_kv_used = 0

    for i in range(lut.shape[0]):
        num_kv_used += count_distinct_elements_per_row(lut[i, -1:]).sum().item()

    return num_kv_used / num_total

def lut_attention_density(
    lut: List[torch.IntTensor],
    block_size: int = 64
):
    if isinstance(lut, str):
        lut = torch.load(lut)

    density_list = []
    for i in range(len(lut)):
        density_list.append(lut_single_layer_attention_density(lut[i], block_size))

    return density_list, sum(density_list) / len(density_list)

def lut_kv_cache_density(
    lut: List[torch.IntTensor],
    block_size: int = 64
):
    if isinstance(lut, str):
        lut = torch.load(lut)

    density_list = []
    for i in range(len(lut)):
        density_list.append(lut_single_layer_kv_cache_density(lut[i], block_size))

    return density_list, sum(density_list) / len(density_list)



class RetinalVesselDataset(Dataset):
    """Retinal Vessel Segmentation Dataset"""
    
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(512, 512)):
        """
        Parameters:
            images_dir (str): Path to images directory
            masks_dir (str): Path to mask labels directory
            transform (callable, optional): Optional transform to be applied to samples
            target_size (tuple): Target size to resize images and masks
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get all image filenames (without path and extension)
        self.img_names = [f.split('.')[0] for f in os.listdir(images_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, f"{img_name}.png")
        img = Image.open(img_path).convert("RGB")
        
        # Load mask (label)
        mask_path = os.path.join(self.masks_dir, f"{img_name}.png")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        
        # Resize image and mask to the same size
        img = img.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)
        
        # Convert mask to binary mask (0 or 1)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)  # Binarize mask
        
        # Apply transforms (if any)
        if self.transform:
            img = self.transform(img)
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        else:
            # Default conversion to tensor
            img = transforms.ToTensor()(img)
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
            
        return {
            'image': img,
            'mask': mask,
            'name': img_name
        }

def get_data_loaders(data_dir, batch_size=4, val_split=0.2, transform=None, target_size=(512, 512)):
    """
    Create training and validation data loaders
    
    Parameters:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size
        val_split (float): Proportion of validation set
        transform (callable, optional): Optional transform to be applied to samples
        target_size (tuple): Target size to resize images and masks
        
    Returns:
        train_loader, val_loader: Training and validation data loaders
    """
    
    # Set default transforms
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset
    dataset = RetinalVesselDataset(
        images_dir=os.path.join(data_dir, 'Training_Images'),
        masks_dir=os.path.join(data_dir, 'Training_Labels'),
        transform=transform,
        target_size=target_size
    )
    
    # Split into training and validation sets
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

def get_test_loader(data_dir, batch_size=1, transform=None, target_size=(512, 512)):
    """
    Create test data loader
    
    Parameters:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size
        transform (callable, optional): Optional transform to be applied to samples
        target_size (tuple): Target size to resize images and masks
        
    Returns:
        test_loader: Test data loader
    """
    
    # Set default transforms
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset
    test_dataset = RetinalVesselDataset(
        images_dir=os.path.join(data_dir, 'Test_Images'),
        masks_dir=os.path.join(data_dir, 'Test_Labels'),
        transform=transform,
        target_size=target_size
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return test_loader 