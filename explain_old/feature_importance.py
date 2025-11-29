"""
explain/feature_importance.py

High-level driver to compute per-image feature importance heatmaps.
Supports multiple explanation methods.
"""

# import os
# import cv2
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Optional

# from explain.explain_utils import produce_map_from_grad_or_compute
# from lvq.model import GrassmannLVQModel


# # helper for saving
# def save_map_numpy(map_hw: torch.Tensor, path: str):
#     arr = map_hw.detach().cpu().numpy()
#     arr = arr - arr.min()
#     arr = arr / (arr.max() + 1e-12)
#     heatmap = cv2.applyColorMap(np.uint8(255 * arr), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
#     plt.imsave(path, heatmap, vmin=0, vmax=1)
#     return heatmap


import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from explain_old.importance_scores import compute_feature_importance, save_feature_importance_heatmap
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

        compute_single_image_heatmap(
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
    import numpy as np
    
    image_resized = F.resize(original_image, (args.image_size, args.image_size)) 
    image_resized = np.array(image_resized).astype(np.float32) 
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
    
        region_heatmap, region_heatmap_per_principal_dir = compute_feature_importance(
            feature, label, Rt, S, output,
            model.prototype_layer.xprotos,
            # model.prototype_layer.yprotos_mat,
            # model.prototype_layer.yprotos_comp_mat,
            model.prototype_layer.relevances,
            # return_full_output=False
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

    overlay = 0.5 * image_resized / 255 + 0.3 * heatmap_upsampled_normalized
    OVERLAY_PATH = os.path.join(out_dir, 'heatmap_original_image.png')

    plt.imsave(fname=OVERLAY_PATH, arr=overlay, vmin=0.0, vmax=1.0)


    logger.info(f"The image with its heatmap has been saved in '{OVERLAY_PATH}'.\n")

    # region_effect_maps_per_principal_direction.append(
    #     region_heatmap_per_principal_dir
    # )
    #################
    
    # Compute feature importance gradient
    # Since nprotos = nclasses, we can use label as prototype index
    # sim_grad = compute_feature_importance(
    #     feature, 
    #     Rt, 
    #     S, 
    #     output,
    #     model.prototype_layer.xprotos,
    #     model.prototype_layer.relevances,
    #     target_class=int(label.item())  # Get gradient for the true class
    # )

    # sim_grad is now (C, H, W) for the target class
    # Aggregate over channels to get spatial importance map
    # importance_map = torch.norm(sim_grad, dim=0)  # (H, W)
    return None
    
    # Upsample to original image size
    importance_map_upsampled = torch.nn.functional.interpolate(
        importance_map.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        size=(args.image_size, args.image_size),
        mode='bilinear',
        align_corners=False
    ).squeeze()  # (image_size, image_size)
    
    # Save heatmap
    heatmap_path = os.path.join(out_dir, 'heatmap.png')
    heatmap_np = save_feature_importance_heatmap(
        importance_map_upsampled, 
        heatmap_path,
        save_raw=True
    )
    logger.info(f"Saved heatmap to {heatmap_path}")
    
    # Create overlay if original image is available
    if original_image is not None:
        import numpy as np
        # Convert PIL image to numpy array
        img_np = np.array(original_image.resize((args.image_size, args.image_size)))
        if img_np.ndim == 2:  # Grayscale
            img_np = np.stack([img_np] * 3, axis=-1)
        img_rgb = img_np.astype(np.float32) / 255.0
        
        # Create overlay
        overlay = 0.5 * img_rgb + 0.5 * heatmap_np
        overlay_path = os.path.join(out_dir, 'overlay.png')
        plt.imsave(overlay_path, overlay, vmin=0, vmax=1)
        logger.info(f"Saved overlay to {overlay_path}")
    
    # Also compute and save gradients for all prototypes (optional)
    if args.__dict__.get('save_all_prototypes', False):
        all_grads = compute_feature_importance(
            feature, 
            Rt, 
            S, 
            output,
            model.prototype_layer.xprotos,
            model.prototype_layer.relevances,
            target_class=None  # Get all prototypes
        )
        # all_grads is (nprotos, C, H, W)
        for proto_idx in range(all_grads.shape[0]):
            proto_importance = torch.norm(all_grads[proto_idx], dim=0)  # (H, W)
            proto_importance_up = torch.nn.functional.interpolate(
                proto_importance.unsqueeze(0).unsqueeze(0),
                size=(args.image_size, args.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            proto_heatmap_path = os.path.join(out_dir, f'heatmap_proto_{proto_idx}.png')
            save_feature_importance_heatmap(proto_importance_up, proto_heatmap_path)
        
        logger.info(f"Saved heatmaps for all {all_grads.shape[0]} prototypes")











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



# def compute_feature_importance_heatmap(model: GrassmannLVQModel,
#                  img_names: List,
#                  imgs_transformed: torch.Tensor,
#                  labels,
#                  logger,
#                  args,
#                 #  loss_fn
#                  ):
    
#     img_size = args.__dict__.get('image_size', 224)

#     OUTPUT_DIR = args.results_dir

#     region_effect_maps_per_principal_direction = []
#     images_resized = []
#     for img_name, label, sample in zip(img_names, labels, imgs_transformed):

#         fname = os.path.splitext(img_name)[0]
#         print("\n", fname)

#         INPUT_PATH = os.path.join(OUTPUT_DIR, fname, img_name)
#         HEATMAP_PATH = os.path.join(OUTPUT_DIR, fname, 'heatmap.png')

#         image = cv2.imread(INPUT_PATH)

#         image_resized = cv2.resize(image, (img_size, img_size),
#                                    interpolation=cv2.INTER_LINEAR)

#         images_resized.append(image)

#         with (torch.no_grad()):
#             feature, subspace, Rt, S, output = model.forward_partial(sample.unsqueeze(0))
#             # print("features", feature.shape, subspace.shape, S.shape, Rt.shape, output['Q'].shape, output['Qw'].shape)
#             # print(torch.diag(S[0]))

#             # region_heatmap, region_heatmap_per_principal_dir = 

#             region_heatmaps, region_heatmaps_times_feature_map = compute_feature_importance(
#                 feature, Rt, S, output,
#                 model.prototype_layer.xprotos,
#                 model.prototype_layer.relevances,
#             )
            
#         # region_heatmap = region_heatmaps_times_feature_map[label].sum(axis=0)
#         region_heatmap = region_heatmaps[label].sum(axis=0)
#         print(region_heatmap.shape)
#         save_feature_importance_heatmap(region_heatmap, output_path=HEATMAP_PATH)
#         logger.info(f"The importance of regions (of '{img_name}') has been completed!")
#         logger.info(f"Its heatmap has been saved in '{HEATMAP_PATH}'.")

#         # Resize to image size and save the (upsampled) heatmap
#         heatmap_upsampled = cv2.resize(
#             region_heatmap.numpy(),
#             dsize=(img_size, img_size), #(sample_array.shape[1], sample_array.shape[0]),
#             interpolation=cv2.INTER_CUBIC
#         )

#         UPSAMPLED_HEATMAP_PATH = os.path.join(OUTPUT_DIR, fname, 'heatmap_upsampled.png')
#         heatmap_upsampled_normalized = save_feature_importance_heatmap(heatmap_upsampled, UPSAMPLED_HEATMAP_PATH)

#         overlay = 0.5 * image_resized / 255 + 0.3 * heatmap_upsampled_normalized
#         OVERLAY_PATH = os.path.join(OUTPUT_DIR, fname, 'heatmap_original_image.png')

#         plt.imsave(fname=OVERLAY_PATH, arr=overlay, vmin=0.0, vmax=1.0)


#         logger.info(f"The image with its heatmap has been saved in '{OVERLAY_PATH}'.\n")



