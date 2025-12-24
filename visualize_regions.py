import argparse
from explain.regions.visualization import visualize_regions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize region-based explanation results in a grid.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the result folder of a single image.")
    parser.add_argument("--output_name", type=str, default="summary_visualization.png", help="Name of the output grid image.")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid.")
    
    args = parser.parse_args()
    visualize_regions(args.input_dir, args.output_name, args.cols)
