import os
import shutil
import random
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def select_random_images(src_dir, dest_dir):
    """
    Selects a random image from each folder inside src_dir,
    renames it based on the folder name, and copies it to dest_dir.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for folder in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder)
        if os.path.isdir(folder_path):
            images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            if images:
                selected_image = random.choice(images)
                src_image_path = os.path.join(folder_path, selected_image)
                dest_image_path = os.path.join(dest_dir, f"{folder}{os.path.splitext(selected_image)[1]}")
                shutil.copy(src_image_path, dest_image_path)
                print(f"Copied {selected_image} as {folder}{os.path.splitext(selected_image)[1]}")


def extract_features(image_folder, model, device='cpu'):
    """
    Extract features from images using a pre-trained model.

    Parameters:
    -----------
    image_folder : str
        Path to the folder containing images.
    model : nn.Module
        Pre-trained model for feature extraction.
    device : str, optional
        Device to use (default is 'cpu').

    Returns:
    --------
    tuple
        - features (Tensor): Extracted features of shape (num_images, feature_dim).
        - labels (Tensor): Corresponding class labels.
    """
    img_size=224
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    model = model.to(device)
    # model = nn.Sequential(*list(model.children())[:-1])  # Keep up to the second-to-last layer
    model.eval()

    features, labels = [], []

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if os.path.isfile(image_path):
            label = image_name.split('.')[0]  # Extract label from filename
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model(image)

            features.append(feature.squeeze().cpu())
            labels.append(int(label) - 1)

    return torch.stack(features), torch.tensor(labels)


def init_prototypes_from_images(image_folder, model, dim_of_subspace, device='cpu'):
    """
    Initialize prototypes using extracted features.

    Parameters:
    -----------
    image_folder : str
        Path to the folder containing images.
    model : nn.Module
        Pre-trained model for feature extraction.
    dim_of_subspace : int
        Dimensionality of the subspace.
    device : str, optional
        Device to use (default is 'cpu').

    Returns:
    --------
    tuple
        - xprotos (Tensor): Initialized prototypes of shape (num_classes, feature_dim, dim_of_subspace).
        - yprotos (Tensor): Labels of the prototypes.
        - yprotos_mat (Tensor): One-hot encoded labels.
        - yprotos_mat_comp (Tensor): Complementary one-hot encoded labels.
    """
    features, labels = extract_features(image_folder, model, device)
    num_classes = len(torch.unique(labels))

    prototype_shape = (num_classes, features.shape[1], dim_of_subspace)
    Q, _ = torch.linalg.qr(features, mode='reduced')
    # Q, _ = torch.linalg.qr(0.5 + 0.1 * torch.randn(prototype_shape, device=device), mode='reduced')
    xprotos = nn.Parameter(Q)

    yprotos_mat = torch.zeros((num_classes, num_classes), dtype=torch.int32, device=device)
    yprotos_mat_comp = torch.ones((num_classes, num_classes), dtype=torch.int32, device=device)

    for i, class_label in enumerate(labels):
        yprotos_mat[class_label, i] = 1
        yprotos_mat_comp[class_label, i] = 0

    return xprotos, labels.to(device), yprotos_mat.to(device), yprotos_mat_comp.to(device)


# Example usage
src_directory = "path/to/your/training_set"  # Change this to your dataset folder
dest_directory = "path/to/your/images"  # Change this to your target folder
select_random_images(src_directory, dest_directory)

# Load a pre-trained model (e.g., ResNet without the classifier layer)
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer

dim_of_subspace = 10  # Example subspace dimension
prototypes = init_prototypes_from_images(dest_directory, model, dim_of_subspace)
