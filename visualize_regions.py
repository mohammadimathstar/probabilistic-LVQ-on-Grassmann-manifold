import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import math

def visualize_regions(input_dir: str, output_name: str = "summary_visualization.png", cols: int = 4):
    """
    Create a grid visualization of region-based explanation results.
    """
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
    
    # Add total heatmap
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
            titles.append(f"Direction {idx + 1}")
            
    if not images_to_plot:
        print(f"No images found in {input_dir}")
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
            # We keep the axis on but remove ticks to show the frame
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
    
    print(f"Summary visualization saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize region-based explanation results in a grid.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the result folder of a single image.")
    parser.add_argument("--output_name", type=str, default="summary_visualization.png", help="Name of the output grid image.")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid.")
    
    args = parser.parse_args()
    visualize_regions(args.input_dir, args.output_name, args.cols)
