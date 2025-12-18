import logging

import torch
import torchvision.transforms as transforms

import argparse
import os
from PIL import Image
import pickle
import cv2


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
    img_size = args.__dict__.get('image_size', 224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_pipeline = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
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
    elif args.dataset == 'MURA':
        return filename.split("_")[2].split("-")[0]
    elif args.dataset == 'SkinCancerISIC':
        return "_".join(filename.split("_")[:-2])
    else:
        raise ValueError(f"Invalid dataset name provided: {args.dataset}")


def get_image_files(args: argparse.Namespace, class_mapping: dict):
    """
    Get list of image files with their labels.
    
    Returns:
        list of tuples: (filename, label_index, class_name)
    """
    images_dir = os.path.join(args.sample_dir, args.dataset)
    label_to_index = {label.lower(): idx for idx, label in class_mapping.items()}
    
    image_info_list = []
    
    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
            
        # Extract class name from the image filename
        class_name = extract_class_from_filename(filename, args)
        
        
        # Get the corresponding class index
        label_idx = label_to_index[class_name.lower()]
        
        image_info_list.append((filename, label_idx, class_name))
    
    return image_info_list


def process_single_image(filename: str, 
                        label_idx: int,
                        args: argparse.Namespace,
                        transform: transforms.Compose,
                        logger: logging.Logger):
    """
    Process a single image and save it to the results directory.
    
    Args:
        filename: Image filename
        label_idx: Class label index
        args: Arguments namespace
        transform: Image transformation pipeline
        logger: Logger instance
    
    Returns:
        tuple: (filename, label_tensor, transformed_image_tensor, original_image)
    """
    images_dir = os.path.join(args.sample_dir, args.dataset)
    image_path = os.path.join(images_dir, filename)
    
    # Load image
    image = Image.open(image_path)
    if args.dataset == 'MURA':
        image = image.convert("RGB")
    
    # Transform image
    transformed_image = transform(image)
    # image_resized = cv2.resize(image, (args.image_size, args.image_size),
    #                                interpolation=cv2.INTER_LINEAR)
    
    # Save the original image in the results folder
    image_basename = os.path.splitext(filename)[0]
    destination_folder = os.path.join(args.results_dir, image_basename)
    os.makedirs(destination_folder, exist_ok=True)
    image.save(os.path.join(destination_folder, filename))
    # image_resized.save(os.path.join(destination_folder, f"resized_{filename}"))
    
    # Convert label to tensor
    label_tensor = torch.tensor(label_idx, dtype=torch.int64)
    
    logger.info(f"Loaded image: {filename} (class: {label_idx})")
    
    return filename, label_tensor, transformed_image, image


def load_and_process_images_generator(args: argparse.Namespace, logger: logging.Logger):
    """
    Generator that yields images one-by-one.
    
    Yields:
        tuple: (filename, label_tensor, transformed_image_tensor, original_image)
    """
    class_mapping = load_class_mapping(args)
    transform_pipeline = create_image_transform(args)
    
    # Get all image files
    image_info_list = get_image_files(args, class_mapping)
    
    logger.info(f"Found {len(image_info_list)} images to process.\n")
    
    # Yield images one by one
    for filename, label_idx, class_name in image_info_list:
        yield process_single_image(filename, label_idx, args, transform_pipeline, logger)


