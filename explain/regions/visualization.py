import os
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, Optional
from PIL import Image

import numpy as np

import cv2
import torch
from matplotlib.patches import ConnectionPatch



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
                                                 args: Any,
                                                 patch_base_dir: Optional[str] = None,
                                                 class_name: Optional[str] = None):
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

    # Fixed colors: Red, Green, Blue, Yellow, Pink (BGR for OpenCV)
    colors = [
        (0, 0, 255),   # Red
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
        (0, 255, 255), # Yellow
        (203, 192, 255) # Pink
    ]

    fname = os.path.splitext(img_name)[0]
    result_dir = os.path.join(args.results_dir, args.dataset, fname)
    os.makedirs(result_dir, exist_ok=True)

    # Check if we should use patch features instead of prototypes
    matching_vectors = rotated_prototype.clone()
    if patch_base_dir and class_name:
        for d in range(rotated_prototype.shape[1]):
            feat_path = os.path.join(patch_base_dir, 'closest', class_name, f'direction_{d+1}', 'rank_1_feature.npy')
            if os.path.exists(feat_path):
                patch_feat = np.load(feat_path)
                matching_vectors[:, d] = torch.from_numpy(patch_feat).to(rotated_prototype.device)

    positions_per_dir = k_closest_feature_positions(feature_map, matching_vectors, args.k_nearest)

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


def visualize_regions_with_patch_matching(input_dir: str, 
                                         patch_base_dir: str,
                                         class_name: str,
                                         output_name: str = "summary_with_patch_matching.pdf", 
                                         grid_size: Tuple[int, int] = (7, 7),
                                         relevances: Optional[np.ndarray] = None,
                                         dir_max_values: Optional[List[float]] = None,
                                         total_max: Optional[float] = None):
    """
    Create a grid visualization that connects closest patches to regions in the original image.
    
    Row 1: 
      - Col 0: Image from plot_important_region_per_principal_direction
      - Col 1: Closest Patches (stacked vertically, resized to region size)
      - Col 3: Aggregated Heatmap
    Rows 2+: Per-direction heatmaps (5 per row).
    """

    type_dir = "closest" # closest or important

    # 1. Load images
    original_path = os.path.join(input_dir, "original_image.png")
    highlighted_path = os.path.join(input_dir, "highlighted_regions.png")
    total_heatmap_path = os.path.join(input_dir, "heatmap_original_image.png")
    directions_dir = os.path.join(input_dir, "directions")
    
    if not os.path.exists(original_path):
        return
    
    img_orig = Image.open(original_path).convert('RGB')
    img_orig_np = np.array(img_orig)
    img_h, img_w = img_orig_np.shape[:2]
    
    img_highlighted = Image.open(highlighted_path).convert('RGB') if os.path.exists(highlighted_path) else img_orig
    
    # 2. Load Patches
    patches = []
    patch_titles = []
    subspace_dim = 0
    if relevances is not None:
        subspace_dim = len(relevances)
    else:
        if os.path.exists(directions_dir):
            subspace_dim = len([f for f in os.listdir(directions_dir) if f.startswith("heatmap_overlay_dir_")])

    for d in range(subspace_dim):
        patch_path = os.path.join(patch_base_dir, type_dir, class_name, f'direction_{d+1}', 'rank_1.png')
        if os.path.exists(patch_path):
            patches.append(Image.open(patch_path).convert('RGB'))
            patch_titles.append(f"P{d+1}")
        else:
            patches.append(None)
            patch_titles.append(f"P{d+1} (N/A)")

    # 3. Match Patches to Regions (Cosine Similarity)
    # For each patch, find the region in the original image with highest similarity
    reg_h, reg_w = grid_size
    patch_h, patch_w = img_h // reg_h, img_w // reg_w
    
    matched_positions = []
    for patch in patches:
        if patch is None:
            matched_positions.append(None)
            continue
            
        patch_resized = patch.resize((patch_w, patch_h))
        patch_np = np.array(patch_resized).astype(np.float32).reshape(-1)
        patch_norm = patch_np / (np.linalg.norm(patch_np) + 1e-8)  # Normalize for cosine similarity
        
        max_sim = -1.0
        best_pos = (0, 0)
        
        for r in range(reg_h):
            for c in range(reg_w):
                region = img_orig_np[r*patch_h:(r+1)*patch_h, c*patch_w:(c+1)*patch_w]
                if region.shape[:2] != (patch_h, patch_w):
                    region = cv2.resize(region, (patch_w, patch_h))
                
                region_np = region.astype(np.float32).reshape(-1)
                region_norm = region_np / (np.linalg.norm(region_np) + 1e-8)  # Normalize for cosine similarity
                
                sim = np.dot(patch_norm, region_norm)
                if sim > max_sim:
                    max_sim = sim
                    best_pos = (r, c)
        matched_positions.append(best_pos)
    
    # 3b. Create highlighted image with boxes around regions most similar to each patch
    # Track which patches match which regions to handle overlaps
    region_to_patches = {}
    for patch_idx, pos in enumerate(matched_positions):
        if pos is None:
            continue
        if pos not in region_to_patches:
            region_to_patches[pos] = []
        region_to_patches[pos].append(patch_idx)
    
    img_highlighted_np = img_orig_np.copy()
    
    # Fixed colors: Red, Green, Blue, Yellow, Pink (RGB for PIL/matplotlib)
    colors_rgb = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green  
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 192, 203)   # Pink
    ]
    
    # Draw boxes for each patch's best matching region with offset for overlaps
    for patch_idx, pos in enumerate(matched_positions):
        if pos is None:
            continue
        
        r, c = pos
        color = colors_rgb[patch_idx % len(colors_rgb)]
        
        # Calculate offset based on how many patches have already been drawn for this region
        patches_at_this_region = region_to_patches[pos]
        offset_index = patches_at_this_region.index(patch_idx)
        
        # Base coordinates
        y1, y2 = r * patch_h, (r + 1) * patch_h
        x1, x2 = c * patch_w, (c + 1) * patch_w
        
        # Apply offset (5 pixels per overlap)
        offset = 5 * offset_index
        max_offset = min(patch_w // 4, patch_h // 4)  # Don't offset more than 1/4 of region size
        offset = min(offset, max_offset)
        
        y1_offset = y1 + offset
        y2_offset = y2 - offset
        x1_offset = x1 + offset
        x2_offset = x2 - offset
        
        # Draw thick border (3 pixels)
        thickness = 3
        # Top border
        img_highlighted_np[y1_offset:y1_offset+thickness, x1_offset:x2_offset] = color
        # Bottom border
        img_highlighted_np[y2_offset-thickness:y2_offset, x1_offset:x2_offset] = color
        # Left border
        img_highlighted_np[y1_offset:y2_offset, x1_offset:x1_offset+thickness] = color
        # Right border
        img_highlighted_np[y1_offset:y2_offset, x2_offset-thickness:x2_offset] = color
    
    # Add number labels using PIL for better text rendering
    from PIL import ImageDraw, ImageFont
    img_pil = Image.fromarray(img_highlighted_np)
    draw = ImageDraw.Draw(img_pil)
    
    # Use a larger font size for visibility
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw number labels for each patch
    for patch_idx, pos in enumerate(matched_positions):
        if pos is None:
            continue
        
        r, c = pos
        color = colors_rgb[patch_idx % len(colors_rgb)]
        
        # Calculate same offset as before
        patches_at_this_region = region_to_patches[pos]
        offset_index = patches_at_this_region.index(patch_idx)
        offset = 5 * offset_index
        max_offset = min(patch_w // 4, patch_h // 4)
        offset = min(offset, max_offset)
        
        y1 = r * patch_h + offset
        x1 = c * patch_w + offset
        
        # Draw text with white background for visibility
        text = str(patch_idx + 1)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # Position in top-left corner with small padding
        text_x = x1 + 5
        text_y = y1 + 5
        
        # Draw white background rectangle
        draw.rectangle([text_x - 2, text_y - 2, text_x + text_w + 2, text_y + text_h + 2], 
                      fill=(255, 255, 255))
        # Draw text in the patch's color
        draw.text((text_x, text_y), text, fill=color, font=font)
    
    img_highlighted_np = np.array(img_pil)
    
    # Update img_highlighted to use our newly created image
    img_highlighted = Image.fromarray(img_highlighted_np)

    # 4. Prepare Heatmaps
    heatmap_imgs = []
    heatmap_titles = []
    if os.path.exists(total_heatmap_path):
        heatmap_imgs.append(Image.open(total_heatmap_path))
        title = "Aggregated Heatmap"
        # Removed total_max from title as per user request
        heatmap_titles.append(title)
        
    if os.path.exists(directions_dir):
        dir_files = sorted([f for f in os.listdir(directions_dir) if f.startswith("heatmap_overlay_dir_")],
                           key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for f in dir_files:
            idx = int(f.split("_")[-1].split(".")[0])
            heatmap_imgs.append(Image.open(os.path.join(directions_dir, f)))
            title = f"Direction {idx + 1}"
            if relevances is not None and idx < len(relevances):
                title += f" (rel: {relevances[idx]:.3f})"
            if dir_max_values is not None and idx < len(dir_max_values):
                title += f" (max: {dir_max_values[idx]:.3f})"
            heatmap_titles.append(title)

    # 5. Create Visualization
    dir_cols = 5
    num_heatmaps = len(heatmap_imgs)
    num_dir_rows = math.ceil((num_heatmaps - 1) / dir_cols) if num_heatmaps > 1 else 0
    total_rows = 1 + num_dir_rows
    max_cols = 5 # Fixed to 5 columns as per user request for heatmap rows
    
    from matplotlib.gridspec import GridSpec
    # Use consistent row height for all rows
    fig = plt.figure(figsize=(max_cols * 3, total_rows * 3))
    gs = GridSpec(total_rows, max_cols, figure=fig, height_ratios=[1] * total_rows)
    
    # Fixed colors matching the first column: Red, Green, Blue, Yellow, Pink
    # Using exact RGB tuples to match OpenCV colors in first column
    # OpenCV uses BGR: (0,0,255)=Red, (0,255,0)=Green, (255,0,0)=Blue, (0,255,255)=Yellow, (203,192,255)=Pink
    # Matplotlib uses RGB, so we convert:
    fixed_colors_rgb = [
        (1.0, 0.0, 0.0),      # Red
        (0.0, 1.0, 0.0),      # Green  
        (0.0, 0.0, 1.0),      # Blue
        (1.0, 1.0, 0.0),      # Yellow
        (1.0, 0.75, 0.8)      # Pink (203/255, 192/255, 255/255 in RGB)
    ]
    
    def add_border(ax):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color('black')
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1, Col 0: Highlighted Image
    ax_high = fig.add_subplot(gs[0, 0])
    ax_high.imshow(img_highlighted)
    ax_high.set_title("Highlighted Regions", fontsize=12, fontweight='bold')
    add_border(ax_high)
    
    # Row 1, Col 1: Patches (Stacked Vertically)
    valid_patches_info = [(d, patches[d]) for d in range(subspace_dim) if patches[d] is not None]
    if valid_patches_info:
        ax_patches = fig.add_subplot(gs[0, 1])
        # Make patches significantly smaller (50% of region size)
        small_patch_w = int(patch_w * 0.5)
        small_patch_h = int(patch_h * 0.5)
        
        # Add spacing between patches (5 pixels white space - 50% of previous)
        spacing = 5
        resized_patches = []
        for _, p in valid_patches_info:
            resized_patch = np.array(p.resize((small_patch_w, small_patch_h)))
            resized_patches.append(resized_patch)
            # Add white spacing after each patch except the last
            if _ != valid_patches_info[-1][0]:
                white_space = np.ones((spacing, small_patch_w, 3), dtype=np.uint8) * 255
                resized_patches.append(white_space)
        
        # Stack them vertically with spacing
        stacked_patches = np.vstack(resized_patches)
        ax_patches.imshow(stacked_patches)
        ax_patches.set_title("Closest Patches", fontsize=12, fontweight='bold')
        # Remove border for patches as per user request
        ax_patches.axis('off')
        
        # Add colored boxes around each patch matching the colors in Column 0
        current_y = 0
        for i, (patch_idx, _) in enumerate(valid_patches_info):
            # Use the same color as the box in Column 0
            color = fixed_colors_rgb[patch_idx % len(fixed_colors_rgb)]
            
            # Draw colored rectangle around patch
            rect = plt.Rectangle((0, current_y), small_patch_w - 1, small_patch_h - 1, 
                                 edgecolor=color, facecolor='none', linewidth=3)
            ax_patches.add_patch(rect)
            
            # Add number label to the right of the patch
            ax_patches.text(small_patch_w + 5, current_y + small_patch_h // 2, 
                           str(patch_idx + 1), 
                           fontsize=14, fontweight='bold', 
                           verticalalignment='center',
                           color=color)
            
            current_y += small_patch_h + spacing
    
    # Row 1, Col 3: Aggregated Heatmap
    if len(heatmap_imgs) > 0:
        ax_agg = fig.add_subplot(gs[0, 3])
        ax_agg.imshow(heatmap_imgs[0])
        ax_agg.set_title(heatmap_titles[0], fontsize=12, fontweight='bold')
        add_border(ax_agg)

    # Rows 2+: Heatmaps (starting from index 1 of heatmap_imgs)
    for i in range(1, num_heatmaps):
        row = 1 + (i - 1) // dir_cols
        col = (i - 1) % dir_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(heatmap_imgs[i])
        ax.set_title(heatmap_titles[i], fontsize=10)
        add_border(ax)

    plt.tight_layout()
    output_path = os.path.join(input_dir, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


