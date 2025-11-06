
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from explain.importance_scores import compute_feature_importance, save_feature_importance_heatmap
import torch

from typing import List

from lvq.model import Model


def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def plot_important_region_per_principal_direction(
        imgs,
        region_importances_per_principal_direction,
        imgs_names,
        args,
        k=1):

    imgW, imgH = 224, 224
    regionW, regionH = region_importances_per_principal_direction.shape[-2:]
    region_size = (int(imgW / regionW), int(imgH / regionH))

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

    for img_name, img, region_map in zip(imgs_names, imgs, region_importances_per_principal_direction):
        result_dir = os.path.join(args.results_dir, img_name[:-4])
        os.makedirs(result_dir, exist_ok=True)

        image = cv2.resize(img, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
        plt.imsave(
            fname=os.path.join(result_dir, 'highlighted_regions.png'),
            arr=image,
            vmin=0.0, vmax=1.0
        )



def compute_feature_importance_heatmap(model: Model,
                 img_names: List,
                 imgs_transformed: torch.Tensor,
                 labels,
                 logger,
                 args):

    OUTPUT_DIR = args.results_dir

    region_effect_maps_per_principal_direction = []
    images_resized = []
    for img_name, label, sample in zip(img_names, labels, imgs_transformed):

        fname = os.path.splitext(img_name)[0]
        print("\n", fname)

        INPUT_PATH = os.path.join(OUTPUT_DIR, fname, img_name)
        HEATMAP_PATH = os.path.join(OUTPUT_DIR, fname, 'heatmap.png')

        image = cv2.imread(INPUT_PATH)

        image_resized = cv2.resize(image, (args.image_size, args.image_size),
                                   interpolation=cv2.INTER_LINEAR)

        images_resized.append(image)

        with (torch.no_grad()):
            feature, subspace, Vh, S, output = model.forward_partial(sample.unsqueeze(0))

            region_heatmap, region_heatmap_per_principal_dir = compute_feature_importance(
                feature, label, Vh, S, output,
                model.prototype_layer.xprotos,
                model.prototype_layer.yprotos_mat,
                model.prototype_layer.yprotos_comp_mat,
                model.prototype_layer.relevances,
                return_full_output=False
            )

        save_feature_importance_heatmap(region_heatmap, output_path=HEATMAP_PATH)
        logger.info(f"The importance of regions (of '{img_name}') has been completed!")
        logger.info(f"Its heatmap has been saved in '{HEATMAP_PATH}'.")

        # Resize to image size and save the (upsampled) heatmap
        heatmap_upsampled = cv2.resize(
            region_heatmap.numpy(),
            dsize=(args.image_size, args.image_size), #(sample_array.shape[1], sample_array.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

        UPSAMPLED_HEATMAP_PATH = os.path.join(OUTPUT_DIR, fname, 'heatmap_upsampled.png')
        heatmap_upsampled_normalized = save_feature_importance_heatmap(heatmap_upsampled, UPSAMPLED_HEATMAP_PATH)

        overlay = 0.5 * image_resized / 255 + 0.3 * heatmap_upsampled_normalized
        OVERLAY_PATH = os.path.join(OUTPUT_DIR, fname, 'heatmap_original_image.png')

        plt.imsave(fname=OVERLAY_PATH, arr=overlay, vmin=0.0, vmax=1.0)


        logger.info(f"The image with its heatmap has been saved in '{OVERLAY_PATH}'.\n")

        region_effect_maps_per_principal_direction.append(
            region_heatmap_per_principal_dir
        )

    return (
        torch.stack(region_effect_maps_per_principal_direction, dim=0),
        images_resized
    )


