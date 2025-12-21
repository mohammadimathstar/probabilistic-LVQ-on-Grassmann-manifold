
import argparse


def get_local_expl_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Explain a prediction')
    parser.add_argument('--model_path',
                        type=str,
                        # default='./run_prototypes/pca/skin/run1/',
                        # default='./run_prototypes/cub/meika/ce/sgd-32batch/d5/run3/',
                        # default='./run_prototypes/cub/meika/reverse/sgd-32batch/dim5/run2/',
                        # default='./run_prototypes/brain/resnet50-v2/reverse-kl/batch32-dim5/run1/',
                        default='./run_prototypes/pca/skin/run1/',
                        help='Directory to trained model')
    parser.add_argument('--which_direction',
                        type=int,
                        default=None,
                        help='Principal direction to consider (0-indexed). If not set, all directions are used.')
    parser.add_argument('--k_negatives',
                        type=int,
                        default=0,
                        help='Number of negative prototypes to consider')
    parser.add_argument('--k_nearest',
                        type=int,
                        default=1,
                        help='Number of nearest feature positions (per principal direction) to consider')
    parser.add_argument('--sample_dir',
                        type=str,
                        default='./samples',
                        help='Directory to image to be explained, or to a folder containing multiple test images')
    parser.add_argument('--results_dir',
                        type=str,
                        default='./explanations',
                        help='Directory where explanations will be saved')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    # parser.add_argument('--image_size',
    #                     type=int,
    #                     default=224,
    #                     help='Resize images to this size')
    # parser.add_argument('--epsilon',
    #                 type=float,
    #                 default=1e-3,
    #                 help='Epsilon value for FDivergence loss function')
    # parser.add_argument('--beta',
    #                 type=float,
    #                 default=1.0,
    #                 help='Beta value for the similarity measure.')
    args = parser.parse_args()
    return args
