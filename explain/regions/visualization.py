import os
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, Optional
from PIL import Image

import numpy as np

import cv2
import torch



def k_closest_feature_positions(feature_map: torch.Tensor, 
                               prototype_vectors: torch.Tensor, 
                               k: int = 1) -> List[List[Tuple[int, int]]]:
    """
    Find the k spatial positions in the feature map that are most similar (cosine similarity)
    to each column (direction) of the prototype vectors.

    Args:
        feature_map: Tensor of shape (C, H, W)
        prototype_vectors: Tensor of shape (C, d)
        k: Number of closest positions to return per direction

    Returns:
        A list of length d, where each item is a list of k (row, col) tuples.
    """
    C, H, W = feature_map.shape
    reshaped_fm = feature_map.view(C, H * W)

    # Normalize for cosine similarity
    fm_norm = torch.nn.functional.normalize(reshaped_fm, dim=0)
    proto_norm = torch.nn.functional.normalize(prototype_vectors, dim=0)

    # Compute similarity matrix (H*W, d)
    similarities = fm_norm.T @ proto_norm
    _, topk_indices = torch.topk(similarities, k=k, dim=0)

    results = []
    for d in range(prototype_vectors.shape[1]):
        indices = topk_indices[:, d]
        positions = [(idx.item() // W, idx.item() % W) for idx in indices]
        results.append(positions)

    return results


def plot_important_region_per_principal_direction(image: np.ndarray,
                                                 feature_map: torch.Tensor,
                                                 rotated_prototype: torch.Tensor,
                                                 img_name: str,
                                                 args: Any):
    """
    Visualize the k-closest feature positions for each principal direction by drawing
    colored rectangles on the original image.
    """
    img_h, img_w = image.shape[:2]
    reg_h, reg_w = feature_map.shape[-2:]
    
    region_w = img_w // reg_w
    region_h = img_h // reg_h

    # Convert to uint8 for OpenCV
    draw_img = (image * 255).astype(np.uint8).copy()

    # BGR colors for OpenCV
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (0, 128, 255), (0, 0, 128),
        (128, 128, 0)
    ]

    fname = os.path.splitext(img_name)[0]
    result_dir = os.path.join(args.results_dir, args.dataset, fname)
    os.makedirs(result_dir, exist_ok=True)

    positions_per_dir = k_closest_feature_positions(feature_map, rotated_prototype, args.k_nearest)

    selected_regions = {}
    for d_idx in range(rotated_prototype.shape[1]):
        positions = positions_per_dir[d_idx]
        color = colors[d_idx % len(colors)]
        
        for r_idx, c_idx in positions:
            key = (r_idx, c_idx)
            selected_regions.setdefault(key, []).append(d_idx)
            
            # Offset to handle overlapping rectangles
            offset = 3 * (len(selected_regions[key]) - 1)
            max_off_w = (region_w - 2) // 2
            max_off_h = (region_h - 2) // 2
            offset = max(0, min(offset, max_off_w, max_off_h))

            start = (c_idx * region_w + offset, r_idx * region_h + offset)
            end = ((c_idx + 1) * region_w - offset, (r_idx + 1) * region_h - offset)
            cv2.rectangle(draw_img, start, end, color, thickness=2)
    
    # Draw legend
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(rotated_prototype.shape[1]):
        color = colors[i % len(colors)]
        y_pos = 20 + i * 15
        cv2.rectangle(draw_img, (10, y_pos - 7), (25, y_pos + 5), color, -1)
        cv2.putText(draw_img, str(i + 1), (30, y_pos + 4), font, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

    # Save result
    highlight_path = os.path.join(result_dir, 'highlighted_regions.png')
    plt.imsave(highlight_path, draw_img.astype(np.float32) / 255.0)


def visualize_regions(input_dir: str, 
                      output_name: str = "summary_visualization.pdf", 
                      cols: int = 4,
                      relevances: Optional[np.ndarray] = None):
    """
    Create a grid visualization of region-based explanation results.
    """
    relevances = None
    
    # 1. Identify images to include
    original_path = os.path.join(input_dir, "original_image.png")
    total_heatmap_path = os.path.join(input_dir, "heatmap_original_image.png")
    directions_dir = os.path.join(input_dir, "directions")
    
    images_to_plot = []
    titles = []
    
    # Add original image
    if os.path.exists(original_path):
        images_to_plot.append(Image.open(original_path))
        titles.append("Original Image")
    
    # Add aggregate attribution heatmap
    if os.path.exists(total_heatmap_path):
        images_to_plot.append(Image.open(total_heatmap_path))
        titles.append("Aggregate Attribution")
    
    # Add per-direction heatmaps
    if os.path.exists(directions_dir):
        # Sort direction heatmaps by index
        dir_files = sorted([f for f in os.listdir(directions_dir) if f.startswith("heatmap_overlay_dir_")],
                           key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for f in dir_files:
            idx = int(f.split("_")[-1].split(".")[0])
            images_to_plot.append(Image.open(os.path.join(directions_dir, f)))
            
            title = f"Direction {idx + 1}"
            if relevances is not None and idx < len(relevances):
                title += f" (rel: {relevances[idx]:.3f})"
            titles.append(title)
            
    if not images_to_plot:
        return

    # 2. Calculate grid layout
    num_images = len(images_to_plot)
    rows = math.ceil(num_images / cols)
    
    # 3. Create plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i in range(len(axes)):
        ax = axes[i]
        if i < num_images:
            ax.imshow(images_to_plot[i])
            ax.set_title(titles[i], fontsize=16, fontweight='bold', pad=10)
            
            # Add a box around the image
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color('black')
        else:
            ax.axis('off') # Hide empty slots
            
    plt.tight_layout(pad=1.5)
    
    # 4. Save result
    output_path = os.path.join(input_dir, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


