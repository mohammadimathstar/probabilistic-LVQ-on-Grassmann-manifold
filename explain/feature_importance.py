"""
explain/feature_importance.py

High-level driver to compute per-image feature importance heatmaps.
Supports multiple explanation methods.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional



from explain.importance_scores import compute_feature_importance, save_feature_importance_heatmap
import torch

from typing import List

from lvq.model import GrassmannLVQModel
import torchvision.transforms.functional as F

# from explain.explain_utils import produce_map_from_grad_or_compute

# helper for saving
def save_map_numpy(map_hw: torch.Tensor, path: str):
    arr = map_hw.detach().cpu().numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-12)
    heatmap = cv2.applyColorMap(np.uint8(255 * arr), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    plt.imsave(path, heatmap, vmin=0, vmax=1)
    return heatmap


def compute_feature_importance_heatmap(
    model,
    image_generator,
    # img_names: List[str],
    # imgs_transformed: torch.Tensor,
    # labels,
    logger,
    args,
    # method: str = 'raw_grad',
    # grad_provided: Optional[List[torch.Tensor]] = None,
):
    """
    Compute feature importance heatmaps for all images using a generator.
    
    Args:
        model: GrassmannLVQ model
        image_generator: Generator yielding (filename, label, transformed_image, original_image)
        logger: Logger instance
        args: Arguments namespace
    """
    os.makedirs(args.results_dir, exist_ok=True)
    
    processed_count = 0
    for img_name, label, img_transformed, original_image in image_generator:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing image {processed_count + 1}: {img_name}")
        logger.info(f"{'='*60}")

        region_importance_per_principal_dir = compute_single_image_heatmap(
            model=model,
            img_name=img_name,
            img_transformed=img_transformed,
            label=label,
            logger=logger,
            args=args,
            original_image=original_image
        )
        
        
        processed_count += 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed processing {processed_count} images")
    logger.info(f"{'='*60}\n")


def compute_single_image_heatmap(
    model,
    img_name: str,
    img_transformed: torch.Tensor,
    label: torch.Tensor,
    logger,
    args,
    original_image=None,
):
    """
    Compute and save heatmap for a single image.
    
    Args:
        model: GrassmannLVQ model
        img_name: Image filename
        img_transformed: Transformed image tensor (C, H, W)
        label: Ground truth label (scalar tensor)
        logger: Logger instance
        args: Arguments namespace
        original_image: Original PIL image for overlay (optional)
    """
    fname = os.path.splitext(img_name)[0]
    out_dir = os.path.join(args.results_dir, fname)
    os.makedirs(out_dir, exist_ok=True)
    
    
    image_resized = F.resize(original_image, (args.image_size, args.image_size)) 
    image_resized_np = np.array(image_resized).astype(np.float32) / 255.0
    # cv2.resize(
    #     original_image,
    #     dsize=(args.image_size, args.image_size),
    #     interpolation=cv2.INTER_CUBIC
    # )
    
    # Add batch dimension for model forward pass
    sample = img_transformed.unsqueeze(0)  # (1, C, H, W)

    # Get model outputs
    with torch.no_grad():
        feature, subspace, Rt, S, output = model.forward_partial(sample)

    # feature = feature.squeeze(0)  # C,H,W
    
        region_heatmap, rotated_prototype_pos = compute_feature_importance(
            feature, label, Rt, S, output,
            model.prototype_layer.xprotos,
            model.prototype_layer.relevances,
            k_negatives=args.k_negatives    
        )

    #######################
    HEATMAP_PATH = os.path.join(out_dir, 'heatmap.png')
    save_feature_importance_heatmap(region_heatmap, output_path=HEATMAP_PATH)
    logger.info(f"The importance of regions (of '{img_name}') has been completed!")
    logger.info(f"Its heatmap has been saved in '{HEATMAP_PATH}'.")

    # Resize to image size and save the (upsampled) heatmap
    heatmap_upsampled = cv2.resize(
        region_heatmap.numpy(),
        dsize=(args.image_size, args.image_size), #(sample_array.shape[1], sample_array.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    UPSAMPLED_HEATMAP_PATH = os.path.join(out_dir, 'heatmap_upsampled.png')
    heatmap_upsampled_normalized = save_feature_importance_heatmap(heatmap_upsampled, UPSAMPLED_HEATMAP_PATH)

    # Use 0.6/0.4 ratio for better visibility
    overlay = 0.6 * image_resized_np  + 0.4 * heatmap_upsampled_normalized
    OVERLAY_PATH = os.path.join(out_dir, 'heatmap_original_image.png')

    plt.imsave(fname=OVERLAY_PATH, arr=overlay, vmin=0.0, vmax=1.0)


    logger.info(f"The image with its heatmap has been saved in '{OVERLAY_PATH}'.\n")

    plot_important_region_per_principal_direction(
            image_resized_np, feature[0], rotated_prototype_pos, img_name, args)





def k_closest_feature_positions(feature_map, V, k=1):
    """
    feature_map: tensor of shape (C, H, W)
    V:           tensor of shape (C, d)
    k:           number of closest positions per direction

    Returns:
        A list (length d), each item is a list of k (h, w) tuples
    """

    C, H, W = feature_map.shape
    HW = H * W

    # reshape feature map to (C, HW)
    reshaped_feature_map = feature_map.view(C, HW)

    # normalize for cosine similarity
    F_norm = torch.nn.functional.normalize(reshaped_feature_map, dim=0)  # (C, HW)
    V_norm = torch.nn.functional.normalize(V, dim=0)                      # (C, d)

    # cosine similarity matrix → (HW, d)
    # each column j gives sim(feature_map[:, i], V[:, j])
    sims = F_norm.T @ V_norm   # (HW, d)

    # for "closest" vectors, we want **largest cosine similarity**
    topk_vals, topk_idx = torch.topk(sims, k=k, dim=0)  # (k, d)

    # convert flat HW indices → (h, w)
    results = []
    for j in range(V.shape[1]):     # loop over each direction
        indices = topk_idx[:, j]    # (k,)
        positions = [(idx.item() // W, idx.item() % W) for idx in indices]
        results.append(positions)

    return results




def plot_important_region_per_principal_direction(
        image,
        feature_map,
        rotated_prototype,
        img_name,
        args):

    # CHECK REGION_MAP_SHAPE: it should be 
    imgH, imgW = image.shape[:2]  # image size (H, W)

    regionH, regionW = feature_map.shape[-2:]  # HxW of region map

    region_size = (
        imgW // regionW,  # width of each region in pixels
        imgH // regionH   # height of each region in pixels
    )
    

    # convert image [0,1] float32 -> uint8 for OpenCV drawing
    image_uint8 = (image * 255).astype(np.uint8)
    draw_img = image_uint8.copy()  # draw on this copy

    # Predefined color palette (BGR for OpenCV)
    color_palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (0, 128, 255), (0, 0, 128),
        (128, 128, 0)
    ]

    result_dir = os.path.join(args.results_dir, img_name[:-4])
    os.makedirs(result_dir, exist_ok=True)

    # Save original image
    plt.imsave(
        fname=os.path.join(result_dir, 'original.png'),
        arr=image,
        vmin=0.0, vmax=1.0
    )
    positions_per_direction = k_closest_feature_positions(feature_map, rotated_prototype, args.k_nearest)

    selected_regions = {}
    for direction_idx in range(rotated_prototype.shape[1]):     # V has shape (C, d)
        ids = positions_per_direction[direction_idx]   

        color = color_palette[direction_idx % len(color_palette)]
        for row_idx, col_idx in ids:
            key = (row_idx, col_idx)
            selected_regions.setdefault(key, []).append(direction_idx)

            offset = 3 * (len(selected_regions[key])-1) 


            # Cap offset to ensure rectangle has at least 2 pixels width/height
            max_offset_w = (region_size[0] - 2) // 2
            max_offset_h = (region_size[1] - 2) // 2
            offset = min(offset, max_offset_w, max_offset_h)
            offset = max(0, offset)

            start_point = (
                col_idx * region_size[0] + offset,
                row_idx * region_size[1] + offset
            )
            end_point = (
                (col_idx + 1) * region_size[0] - offset,
                (row_idx + 1) * region_size[1] - offset
            )
            draw_img= cv2.rectangle(draw_img, start_point, end_point, color, thickness=2)
    
    # Draw legend (bottom-left corner)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 0.9
    font_thickness = 1
    legend_x, legend_y = 10, 20
    line_spacing = 15

    for i in range(rotated_prototype.shape[1]):
        color = color_palette[i % len(color_palette)]
        label = f"{i+1}"
        cv2.putText(draw_img, label, (legend_x + 20, legend_y + 4 + i * line_spacing),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.rectangle(draw_img, (legend_x, legend_y - 7 + i * line_spacing),
                      (legend_x + 15, legend_y + 5 + i * line_spacing),
                      color, -1)

    # Save final highlighted image
    draw_float = draw_img.astype(np.float32) / 255.0
    plt.imsave(
        fname=os.path.join(result_dir, 'highlighted_regions.png'),
        arr=draw_float,
        vmin=0.0, vmax=1.0
    )



def plot_important_region_per_principal_direction_old(
        image,
        region_map,
        img_name,
        args,
        k=1):

    imgW, imgH = 224, 224
    regionW, regionH = region_map.shape[-2:]
    region_size = (int(imgW / regionW), int(imgH / regionH))

    # image is in float32 [0,1], convert to uint8 for OpenCV drawing
    image_uint8 = (image * 255).astype(np.uint8)

    # Predefined set of distinguishable colors (BGR)
    color_palette = [
        (255, 0, 0),    # 0 - Blue
        (0, 255, 0),    # 1 - Green
        (0, 0, 255),    # 2 - Red
        (255, 255, 0),  # 3 - Cyan
        (255, 0, 255),  # 4 - Magenta
        (0, 255, 255),  # 5 - Yellow
        (128, 0, 128),  # 6 - Purple
        (0, 128, 255),  # 7 - Orange-ish
        (0, 0, 128),    # 8 - Maroon
        (128, 128, 0),  # 9 - Olive
    ]

    result_dir = os.path.join(args.results_dir, img_name[:-4])
    os.makedirs(result_dir, exist_ok=True)

    
    plt.imsave(
            fname=os.path.join(result_dir, 'original.png'),
            arr=image,
            vmin=0.0, vmax=1.0
        )

    # Track which regions were already picked and by which directions
    selected_regions = {}

    for direction_idx, region_per_dir in enumerate(region_map):
        region_per_dir = region_per_dir.numpy()
        ids = k_largest_index_argsort(region_per_dir, k=k)
        color = color_palette[direction_idx % len(color_palette)]

        for row_idx, col_idx in ids:
            key = (row_idx, col_idx)
            selected_regions.setdefault(key, []).append(direction_idx)

            offset = 3 * (len(selected_regions[key]) - 1)  # Offset for overlapping rectangles

            start_point = (
                col_idx * region_size[0] + offset+1,
                row_idx * region_size[1] + offset+1
            )
            end_point = (
                (col_idx + 1) * region_size[0] - offset-1,
                (row_idx + 1) * region_size[1] - offset-1
            )

            image = cv2.rectangle(image, start_point, end_point, color, thickness=2)

    # Draw legend (bottom-left corner)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 0.9
    font_thickness = 1
    legend_x, legend_y = 10, 20
    line_spacing = 15

    for i in range(len(region_map)):
        color = color_palette[i % len(color_palette)]
        label = f"{i+1}"
        cv2.putText(image, label, (legend_x + 20, legend_y+4 + i * line_spacing),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.rectangle(image, (legend_x, legend_y - 7 + i * line_spacing),
                          (legend_x + 15, legend_y + 5 + i * line_spacing),
                          color, -1)

    # final_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float = image_uint8.astype(np.float32) / 255.0
    plt.imsave(
        fname=os.path.join(result_dir, 'highlighted_regions.png'),
        arr=image_float,
        vmin=0.0, vmax=1.0
    )


