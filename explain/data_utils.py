import logging

import torch
import torchvision.transforms as transforms

import argparse
import os
from PIL import Image
import pickle


def load_class_mapping(args: argparse.Namespace) -> dict:
    """
    Load the class index to label mapping from a pickle file.
    """
    mapping_file = os.path.join('./data', f'index2label_{args.dataset}.pkl')
    with open(mapping_file, 'rb') as file:
        class_mapping = pickle.load(file)

    return class_mapping


def create_image_transform(args: argparse.Namespace) -> transforms.Compose:
    """
    Create a preprocessing pipeline for image transformation.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_pipeline = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])

    return transform_pipeline


def extract_class_from_filename(filename: str, args: argparse.Namespace) -> str:
    """
    Extract the class name from the image filename based on the dataset type.
    """
    if args.dataset == 'CARS':
        return filename.split("_")[0]
    elif args.dataset == 'CUB-200-2011':
        return "_".join(filename.split("_")[:-2])
    elif args.dataset == 'BRAIN':
        dic = {'gl': 'glioma', 'me': 'meningioma', 'no': 'notumor', 'pi': 'pituitary'}
        return dic[filename[3:5]]
    elif args.dataset == 'PETS':
        return "_".join(filename.split("_")[:-1])
    else:
        raise ValueError(f"Invalid dataset name provided: {args.dataset}")


def process_images(args: argparse.Namespace,
                   transform: transforms.Compose,
                   class_mapping: dict,
                   logger: logging.Logger):
    """
    Process images by applying transformations and saving results.
    """
    images_dir = os.path.join(args.sample_dir, args.dataset)
    label_to_index = {label.lower(): idx for idx, label in class_mapping.items()}

    processed_images = []
    image_filenames = []
    image_labels = []

    # Iterate through all images in the dataset directory
    for filename in os.listdir(images_dir):
        # Extract class name from the image filename
        class_name = extract_class_from_filename(filename, args)

        # print(class_name)

        # Get the corresponding class index
        image_labels.append(label_to_index[class_name.lower()])

        # Load and transform the image
        image_path = os.path.join(images_dir, filename)
        image = Image.open(image_path)
        processed_images.append(transform(image))
        image_filenames.append(filename)

        # Save the processed image in the appropriate results folder
        image_basename = os.path.splitext(filename)[0]
        destination_folder = os.path.join(args.results_dir, image_basename)
        os.makedirs(destination_folder, exist_ok=True)

        image.save(os.path.join(destination_folder, filename))

    # Stack images into a single tensor
    processed_images = torch.stack(processed_images, dim=0)
    image_labels = torch.tensor(image_labels, dtype=torch.int64)

    logger.info(f"Processed {processed_images.shape[0]} images and saved results to '{args.results_dir}'.\n")

    return image_filenames, image_labels, processed_images


def load_and_process_images(args: argparse.Namespace, logger: logging.Logger):
    """
    Load class mappings, create preprocessing pipeline, and process images.
    """
    class_mapping = load_class_mapping(args)
    transform_pipeline = create_image_transform(args)

    return process_images(args, transform_pipeline, class_mapping, logger)


