"""
explain/utils.py

Utility functions for image processing, data loading, and importance map generation.
"""

import os
import pickle
import logging
from typing import Optional, Tuple, List, Dict, Any, Generator

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# =============================================================================
# Image Processing & Transformation
# =============================================================================

def create_image_transform(image_size: int = 224) -> transforms.Compose:
    """
    Create a standard preprocessing pipeline for images.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def upto_image_size(x_hw: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    Upsample a (H, W) or (1, 1, H, W) tensor to the target image size using bilinear interpolation.
    """
    if x_hw.dim() == 2:
        x = x_hw.unsqueeze(0).unsqueeze(0)
    elif x_hw.dim() == 3 and x_hw.shape[0] == 1:
        x = x_hw.unsqueeze(0)
    else:
        x = x_hw.unsqueeze(0) if x_hw.dim() == 3 else x_hw
    
    x_up = F.interpolate(x, size=img_size, mode='bilinear', align_corners=False)
    return x_up.squeeze(0).squeeze(0)


# =============================================================================
# Data Loading & Dataset Helpers
# =============================================================================

def load_class_mapping(dataset_name: str, data_dir: str = './data') -> Dict[int, str]:
    """
    Load the class index to label mapping from a pickle file.
    """
    mapping_file = os.path.join(data_dir, f'index2label_{dataset_name}.pkl')
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'rb') as file:
        return pickle.load(file)


def extract_class_from_filename(filename: str, dataset_name: str) -> str:
    """
    Extract the class name from the image filename based on the dataset convention.
    """
    dataset_name = dataset_name.upper()
    if dataset_name == 'CARS':
        return filename.split("_")[0]
    elif dataset_name == 'CUB-200-2011' or dataset_name == 'CUB_200_2011':
        return "_".join(filename.split("_")[:-2])
    elif dataset_name == 'BRAIN':
        dic = {'gl': 'glioma', 'me': 'meningioma', 'no': 'notumor', 'pi': 'pituitary'}
        return dic[filename[3:5]]
    elif dataset_name == 'PETS':
        return "_".join(filename.split("_")[:-1])
    elif dataset_name == 'MURA':
        return filename.split("_")[2].split("-")[0]
    elif dataset_name == 'SKINCANCERISIC':
        return "_".join(filename.split("_")[:-2])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_image_files(sample_dir: str, dataset_name: str, class_mapping: Dict[int, str]) -> List[Tuple[str, int, str]]:
    """
    Get a list of image files with their corresponding labels and class names.
    """
    images_dir = os.path.join(sample_dir, dataset_name)
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
    label_to_index = {label.lower(): idx for idx, label in class_mapping.items()}
    image_info_list = []
    
    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
            
        try:
            class_name = extract_class_from_filename(filename, dataset_name)
            label_idx = label_to_index[class_name.lower()]
            image_info_list.append((filename, label_idx, class_name))
        except (ValueError, KeyError) as e:
            logging.warning(f"Skipping {filename}: {e}")
            
    return image_info_list


def load_and_process_images_generator(args: Any, logger: logging.Logger) -> Generator:
    """
    Generator that yields processed images one-by-one.
    """
    class_mapping = load_class_mapping(args.dataset)
    transform = create_image_transform(args.image_size)
    image_info_list = get_image_files(args.sample_dir, args.dataset, class_mapping)
    
    logger.info(f"Found {len(image_info_list)} images to process in {args.dataset}.\n")
    
    for filename, label_idx, class_name in image_info_list:
        image_path = os.path.join(args.sample_dir, args.dataset, filename)
        image = Image.open(image_path)
        if args.dataset.upper() == 'MURA':
            image = image.convert("RGB")
            
        transformed_image = transform(image)
        label_tensor = torch.tensor(label_idx, dtype=torch.int64)
        
        # Ensure result directory for this image exists
        image_basename = os.path.splitext(filename)[0]
        destination_folder = os.path.join(args.results_dir, args.dataset, image_basename)
        os.makedirs(destination_folder, exist_ok=True)
        
        # Save a copy of the original image
        image.save(os.path.join(destination_folder, filename))
        
        logger.info(f"Loaded image: {filename} (class: {label_idx} - {class_name})")
        yield filename, label_tensor, transformed_image, image


# =============================================================================
# Attribution Map Utilities
# =============================================================================

def raw_grad_map(grad_feat: torch.Tensor) -> torch.Tensor:
    """
    Compute importance map by summing gradients over channels.
    grad_feat: (C, H, W) -> (H, W)
    """
    return grad_feat.sum(dim=0)


def input_x_grad_map(feature_map: torch.Tensor, grad_feat: torch.Tensor) -> torch.Tensor:
    """
    Compute importance map as the absolute sum of (feature_map * grad_feat) over channels.
    """
    prod = feature_map * grad_feat
    return torch.abs(prod.sum(dim=0))


def gradcam_map(feature_map: torch.Tensor, grad_feat: torch.Tensor, relu: bool = True) -> torch.Tensor:
    """
    Compute Grad-CAM map.
    """
    C, H, W = feature_map.shape
    alpha = grad_feat.view(C, -1).mean(dim=1)  # Global average pooling of gradients
    cam = (alpha.view(C, 1, 1) * feature_map).sum(dim=0)
    if relu:
        cam = F.relu(cam)
    return cam
