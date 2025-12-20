
import argparse


def get_local_expl_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Explain a prediction')
    parser.add_argument('--model_path',
                        type=str,
                        # default='./run_prototypes/pca/skin/run1/',
                        # default='./run_prototypes/cub/meika/ce/sgd-32batch/d5/run2/',
                        # default='./run_prototypes/cub/meika/reverse/sgd-32batch/dim5/run2/',
                        # default='./run_prototypes/brain/resnet50-v2/reverse-kl/batch32-dim5/run1/',
                        default='./run_prototypes/pca/cub/run1/',
                        help='Directory to trained model')
    parser.add_argument('--method',
                        type=str,
                        default='smoothgrad',
                        choices=['raw_grad', 'grad_times_input', 'smoothgrad'],
                        help='Feature importance method to use')
    parser.add_argument('--smoothgrad_samples',
                        type=int,
                        default=100,
                        help='Number of negative prototypes to consider')
    parser.add_argument('--smoothgrad_noise_std',
                        type=float,
                        default=0.1,
                        help='Noise standard deviation for SmoothGrad')
    parser.add_argument('--cap_percentile',
                        type=float,
                        default=99.0,
                        help='Number of nearest feature positions (per principal direction) to consider')
    parser.add_argument('--add_log_scaling',
                    type=float,
                    default=None,
                    help='Apply log scaling to saliency map')
    parser.add_argument('--sample_dir',
                        type=str,
                        default='./samples',
                        help='Directory to image to be explained, or to a folder containing multiple test images')
    parser.add_argument('--results_dir',
                        type=str,
                        default='./explanations_grad',
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
