
import argparse


def get_local_expl_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Explain a prediction')
    parser.add_argument('--model',
                        type=str,
                        # default='./run_prototypes/cub200/resnet50-nat/identity/512/10d/fourth-run/checkpoints/best_test_model',
                        # default='./run_prototypes/car/convnext-sigmoid/third/checkpoints/best_test_model',
                        default='./run_prototypes/pets/convnext-sigmoid/512/first/checkpoints/best_test_model',
                        # default='./run_prototypes/brain/resbet-50-v1(identity-i guess)/first-run/checkpoints/best_test_model',
                        # default='./run_prototypes/checkpoints/best_test_model',
                        help='Directory to trained model')
    parser.add_argument('--dataset',
                        type=str,
                        # default='CUB-200-2011',
                        # default='CARS',
                        # default='BRAIN',
                        default='PETS',
                        help='Data set on which the model was trained')
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
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Resize images to this size')
    args = parser.parse_args()
    return args
