import os
import matplotlib.pyplot as plt

import numpy as np

import cv2
import torch



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

