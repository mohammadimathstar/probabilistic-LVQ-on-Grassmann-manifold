import argparse
import yaml
import os
from typing import Any, Dict

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_unified_explain_args() -> argparse.Namespace:
    """
    Get arguments for the explanation process.
    Supports both 'regions' and 'pixels' modes.
    """
    parser = argparse.ArgumentParser(description='Explain a prediction (Regions or Pixels)')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['regions', 'pixels'], default='regions',
                        help='Explanation mode: "regions" (find regions) or "pixels" (find pixels)')
    
    # Config file path (optional, defaults based on mode)
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file')
    
    # Common CLI Overrides
    parser.add_argument('--model_path', type=str, help='Directory to trained model')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--results_dir', type=str, help='Directory where explanations will be saved')
    parser.add_argument('--num_images', type=int, help='Number of images to process')
    parser.add_argument('--disable_cuda', action='store_true', help='Disable GPU usage')

    # Mode-specific CLI Overrides
    parser.add_argument('--explain_method', type=str, help='Method for regions mode (raw_grad, input_x_grad, gradcam)')
    parser.add_argument('--method', type=str, help='Method for pixels mode (raw_grad, smoothgrad, grad_times_input)')

    args = parser.parse_args()

    # Determine config path if not provided
    if not args.config:
        args.config = f'explain/config/{args.mode}_config.yaml'

    # Load YAML config
    config = load_yaml_config(args.config)

    # Merge YAML config with CLI args (CLI takes precedence)
    final_args_dict = config.copy()
    final_args_dict['mode'] = args.mode # Ensure mode is set
    
    for key, value in vars(args).items():
        if value is not None:
            final_args_dict[key] = value

    # Convert back to Namespace for compatibility
    return argparse.Namespace(**final_args_dict)
