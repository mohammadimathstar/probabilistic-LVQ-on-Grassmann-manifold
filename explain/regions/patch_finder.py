import os
import torch
import numpy as np
import cv2
import logging
from tqdm import tqdm
from typing import Any, List, Dict, Tuple
from PIL import Image

def find_closest_patches_from_dataset(model: torch.nn.Module, 
                                     dataloader: torch.utils.data.DataLoader, 
                                     args: Any, 
                                     logger: logging.Logger) -> Dict[int, List[Dict[str, Any]]]:
    """
    Find the closest patches in the dataset to each principal direction of each class prototype.
    """
    model.eval()
    device = next(model.parameters()).device
    
    num_classes = args.nclasses
    subspace_dim = args.dim_of_subspace
    
    # best_patches[class_idx][direction_idx] = {similarity, image_path, row, col}
    best_patches = {c: [None] * subspace_dim for c in range(num_classes)}
    
    # Get prototypes
    with torch.no_grad():
        # xprotos shape: (num_classes, embedding_dim, subspace_dim)
        xprotos = model.prototype_layer.xprotos
    
    logger.info("Starting search for closest patches across the dataset...")
    
    # We need to know the image paths. ImageFolder dataset has 'samples' attribute.
    # However, dataloader might not return paths. Let's assume we can get them from the dataset.
    dataset = dataloader.dataset
    image_paths = [s[0] for s in dataset.samples]
    
    pbar = tqdm(total=len(dataloader), desc="Processing batches")
    
    img_idx_offset = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        
        with torch.no_grad():
            # Forward pass to get feature maps
            # feature shape: (B, C, H, W)
            feature, _, _, _, _ = model.forward_partial(images)
            
            B, C, H, W = feature.shape
            reshaped_fm = feature.view(B, C, H * W) # (B, C, HW)
            
            # Normalize feature maps for cosine similarity
            fm_norm = torch.nn.functional.normalize(reshaped_fm, dim=1) # Normalize over channels
            
            for b in range(B):
                img_idx = img_idx_offset + b
                img_path = image_paths[img_idx]
                label = labels[b].item()
                
                # Prototype for this class
                proto = xprotos[label] # (C, d)
                proto_norm = torch.nn.functional.normalize(proto, dim=0) # (C, d)
                
                # Similarities for this image: (HW, d)
                # fm_norm[b] is (C, HW), so fm_norm[b].T is (HW, C)
                sims = fm_norm[b].T @ proto_norm # (HW, C) @ (C, d) -> (HW, d)
                
                max_sims, max_indices = torch.max(sims, dim=0) # (d,), (d,)
                
                for d in range(subspace_dim):
                    sim_val = max_sims[d].item()
                    if best_patches[label][d] is None or sim_val > best_patches[label][d]['similarity']:
                        idx = max_indices[d].item()
                        best_patches[label][d] = {
                            'similarity': sim_val,
                            'image_path': img_path,
                            'row': idx // W,
                            'col': idx % W,
                            'H': H,
                            'W': W
                        }
        
        img_idx_offset += B
        pbar.update(1)
    
    pbar.close()
    return best_patches

def extract_and_save_patches(best_patches: Dict[int, List[Dict[str, Any]]], 
                             classes: List[str], 
                             args: Any, 
                             logger: logging.Logger):
    """
    Extract patches from original images and save them.
    """
    patch_base_dir = os.path.join(args.results_dir, args.dataset, 'patchs')
    os.makedirs(patch_base_dir, exist_ok=True)
    
    logger.info(f"Saving patches to {patch_base_dir}...")
    logger.info(f"Number of classes in best_patches: {len(best_patches)}")
    logger.info(f"Number of classes in classes list: {len(classes)}")
    
    for class_idx, directions in best_patches.items():
        if class_idx >= len(classes):
            logger.warning(f"class_idx {class_idx} is out of range for classes list (len: {len(classes)}). Skipping.")
            continue
        class_name = classes[class_idx]
        class_dir = os.path.join(patch_base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for d_idx, patch_info in enumerate(directions):
            if patch_info is None:
                continue
            
            img_path = patch_info['image_path']
            row, col = patch_info['row'], patch_info['col']
            H, W = patch_info['H'], patch_info['W']
            
            # Load original image
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size
            
            # Calculate patch coordinates
            # Assuming uniform grid
            patch_w = img_w // W
            patch_h = img_h // H
            
            left = col * patch_w
            top = row * patch_h
            right = (col + 1) * patch_w
            bottom = (row + 1) * patch_h
            
            patch = img.crop((left, top, right, bottom))
            
            save_path = os.path.join(class_dir, f'direction_{d_idx + 1}.png')
            patch.save(save_path)
            
    logger.info("All patches saved successfully.")
