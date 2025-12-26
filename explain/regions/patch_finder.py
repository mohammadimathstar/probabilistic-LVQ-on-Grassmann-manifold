import os
import torch
import numpy as np
import cv2
import logging
from tqdm import tqdm
from typing import Any, List, Dict, Tuple
from PIL import Image

from explain.regions.attribution import compute_feature_importance

def find_closest_patches_from_dataset(model: torch.nn.Module, 
                                     dataloader: torch.utils.data.DataLoader, 
                                     args: Any, 
                                     logger: logging.Logger) -> Dict[str, Dict[int, List[List[Dict[str, Any]]]]]:
    """
    Find the top-K closest patches in the dataset to each principal direction of each class prototype,
    both by cosine similarity and by importance (attribution).
    """
    model.eval()
    device = next(model.parameters()).device
    
    num_classes = args.nclasses
    subspace_dim = args.dim_of_subspace
    k_patches = getattr(args, 'k_nearest_patches', 1)
    
    # results[type][class_idx][direction_idx] = list of {val, image_path, row, col, ...}
    results = {
        'closest': {c: [[] for _ in range(subspace_dim)] for c in range(num_classes)},
        'important': {c: [[] for _ in range(subspace_dim)] for c in range(num_classes)}
    }
    
    # Get prototypes
    with torch.no_grad():
        # xprotos shape: (num_classes, embedding_dim, subspace_dim)
        xprotos = model.prototype_layer.xprotos
        relevances = model.prototype_layer.relevances # (1, subspace_dim)
    
    logger.info(f"Starting search for top-{k_patches} closest and most important patches across the dataset...")
    
    dataset = dataloader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        # Get paths from the underlying dataset using the subset indices
        base_dataset = dataset.dataset
        image_paths = [base_dataset.samples[i][0] for i in dataset.indices]
    else:
        image_paths = [s[0] for s in dataset.samples]
    
    total_images = len(dataset)
    if args.num_images:
        total_images = min(total_images, args.num_images)
    
    pbar = tqdm(total=total_images, desc="Processing images")
    
    img_idx_offset = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        
        # We need to process images one by one for compute_feature_importance
        B = images.shape[0]
        
        for b in range(B):
            img_idx = img_idx_offset + b
            if args.num_images and img_idx >= args.num_images:
                pbar.close()
                return results
            
            img_path = image_paths[img_idx]
            label_idx = labels[b].item()
            
            # Forward pass for this single image
            with torch.no_grad():
                feature, _, Rt, S, output = model.forward_partial(images[b:b+1])
            
            B_f, C, H, W = feature.shape
            reshaped_fm = feature.view(C, H * W) # (C, HW)
            
            # Normalize feature maps for cosine similarity
            # fm_norm = torch.nn.functional.normalize(reshaped_fm, dim=0) ######### Normalize over channels (C)
            fm_norm = reshaped_fm 
            
            # Prototype for this class
            proto = xprotos[label_idx] # (C, d)
            proto_norm = torch.nn.functional.normalize(proto, dim=0) # (C, d)
            
            # 1. Cosine Similarity: (HW, C) @ (C, d) -> (HW, d)
            sims = fm_norm.T @ proto_norm 
            max_sims, max_indices = torch.max(sims, dim=0) # (d,), (d,)
            
            for d in range(subspace_dim):
                sim_val = max_sims[d].item()
                idx = max_indices[d].item()
                
                patch_info = {
                    'similarity': sim_val,
                    'image_path': img_path,
                    'row': idx // W,
                    'col': idx % W,
                    'H': H,
                    'W': W,
                    'feature_vector': reshaped_fm[:, idx].cpu().numpy()
                }
                
                # Maintain top-K
                results['closest'][label_idx][d].append(patch_info)
                results['closest'][label_idx][d].sort(key=lambda x: x['similarity'], reverse=True)
                results['closest'][label_idx][d] = results['closest'][label_idx][d][:k_patches]
            
            # 2. Importance (Attribution)
            for d in range(subspace_dim):
                # Create temporary relevances with only direction d active
                temp_relevances = torch.zeros_like(relevances)
                temp_relevances[0, d] = 1.0
                
                # compute_feature_importance expects batch size 1
                region_heatmap_d, _ = compute_feature_importance(
                    feature, torch.tensor([label_idx], device=device), Rt, S, output,
                    xprotos,
                    temp_relevances,
                    k_negatives=args.k_negatives,
                    args=args,
                    print_info=False
                )
                
                # region_heatmap_d is (H, W)
                imp_val, imp_idx = torch.max(region_heatmap_d.view(-1), dim=0)
                imp_val = imp_val.item()
                imp_idx = imp_idx.item()
                
                patch_info = {
                    'importance': imp_val,
                    'image_path': img_path,
                    'row': imp_idx // W,
                    'col': imp_idx % W,
                    'H': H,
                    'W': W,
                    'feature_vector': reshaped_fm[:, imp_idx].cpu().numpy()
                }
                
                # Maintain top-K
                results['important'][label_idx][d].append(patch_info)
                results['important'][label_idx][d].sort(key=lambda x: x['importance'], reverse=True)
                results['important'][label_idx][d] = results['important'][label_idx][d][:k_patches]
            
            pbar.update(1)
        
        img_idx_offset += B
        if args.num_images and img_idx_offset >= args.num_images:
            break
    
    pbar.close()
    return results

def extract_and_save_patches(best_patches: Dict[str, Dict[int, List[List[Dict[str, Any]]]]], 
                             classes: List[str], 
                             args: Any, 
                             logger: logging.Logger):
    """
    Extract patches from original images and save them.
    """
    patch_base_dir = os.path.join(args.results_dir, args.dataset, 'patchs')
    os.makedirs(patch_base_dir, exist_ok=True)
    
    logger.info(f"Saving patches to {patch_base_dir}...")
    
    for patch_type, class_patches in best_patches.items():
        type_dir = os.path.join(patch_base_dir, patch_type)
        os.makedirs(type_dir, exist_ok=True)
        
        for class_idx, directions in class_patches.items():
            if class_idx >= len(classes):
                logger.warning(f"class_idx {class_idx} is out of range for classes list (len: {len(classes)}). Skipping.")
                continue
            class_name = classes[class_idx]
            class_dir = os.path.join(type_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for d_idx, patch_list in enumerate(directions):
                if not patch_list:
                    continue
                
                direction_dir = os.path.join(class_dir, f'direction_{d_idx + 1}')
                os.makedirs(direction_dir, exist_ok=True)
                
                for rank, patch_info in enumerate(patch_list):
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
                    
                    # Filename includes rank
                    save_path = os.path.join(direction_dir, f'rank_{rank + 1}.png')
                    patch.save(save_path)
                    
                    # Save feature vector
                    if 'feature_vector' in patch_info:
                        feat_path = os.path.join(direction_dir, f'rank_{rank + 1}_feature.npy')
                        np.save(feat_path, patch_info['feature_vector'])
                
    logger.info("All patches saved successfully.")
