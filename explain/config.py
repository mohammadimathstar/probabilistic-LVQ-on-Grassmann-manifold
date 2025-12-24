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

def get_explain_args() -> argparse.Namespace:
    """
    Get arguments for the explanation process.
    Loads from YAML by default and allows CLI overrides.
    """
    parser = argparse.ArgumentParser(description='Explain a prediction')
    
    # Config file path
    parser.add_argument('--config', type=str, default='explain/config/explain_config.yaml',
                        help='Path to the YAML configuration file')
    
    # CLI Overrides
    parser.add_argument('--model_path', type=str, help='Directory to trained model')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--explain_method', type=str, choices=['raw_grad', 'input_x_grad', 'gradcam'],
                        help='Method to compute importance scores')
    parser.add_argument('--results_dir', type=str, help='Directory where explanations will be saved')
    parser.add_argument('--num_images', type=int, help='Number of images to process')
    parser.add_argument('--disable_cuda', action='store_true', help='Disable GPU usage')

    args = parser.parse_args()

    # Load YAML config
    config = load_yaml_config(args.config)

    # Merge YAML config with CLI args (CLI takes precedence)
    final_args_dict = config.copy()
    for key, value in vars(args).items():
        if value is not None or key not in final_args_dict:
            final_args_dict[key] = value

    # Convert back to Namespace for compatibility
    return argparse.Namespace(**final_args_dict)
