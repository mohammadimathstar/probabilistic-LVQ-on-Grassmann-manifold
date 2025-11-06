
import numpy as np
import argparse
import os
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda

from typing import Tuple, Dict
from torch import Tensor
import random
from sklearn.model_selection import train_test_split


def get_data(args: argparse.Namespace): 
    """
    Load the proper dataset based on the parsed arguments
    :param args: The arguments in which is specified which dataset should be used
    :return: a 5-tuple consisting of:
                - The train data set
                - The project data set (usually train data set without augmentation)
                - The test data set
                - a tuple containing all possible class labels
                - a tuple containing the shape (depth, width, height) of the input images
    """
    if args.dataset =='CUB-200-2011':
        return get_birds(True,
                         './data/CUB_200_2011/dataset/train_corners',
                         './data/CUB_200_2011/dataset/train_crop',
                         './data/CUB_200_2011/dataset/test_full')
    if args.dataset == 'CARS':
        return get_cars(True,
                        './data/cars/dataset/train',
                        './data/cars/dataset/train',
                        './data/cars/dataset/test')
    if args.dataset == 'PETS': # from PIPNET
        return get_pets(True,
                        './data/PETS/dataset/train',
                        './data/PETS/dataset/train',
                        './data/PETS/dataset/test')
    if args.dataset == 'BRAIN':
        return get_cars(True,
                        './data/brain-tumor/Training',
                        './data/brain-tumor/Training',
                        './data/brain-tumor/Testing')
    if args.dataset == 'MURA':
        return get_cars(True,
                        './data/MURA-v1.1/dataset/train',
                        './data/MURA-v1.1/dataset/train',
                        './data/MURA-v1.1/dataset/valid')
    if args.dataset == 'ETH-80':
        return get_cars(True,
                        './data/ETH-80/train',
                        './data/ETH-80/train',
                        './data/ETH-80/test')
        # return get_pets(True, './data/PETS/dataset/train', './data/PETS/dataset/train', './data/PETS/dataset/test',
        #                 args.image_size, args.seed, args.validation_size)
        # return get_pets(True, './data/PETS/dataset/train', './data/PETS/dataset/train',
        #                 './data/PETS/dataset/test', # set it None for splitting
        #                 224, args.seed, args.validation_size)
    raise Exception(f'Could not load data set "{args.dataset}"!')

def get_dataloaders(args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, testset, classes, shape = get_data(args)
    c, w, h = shape
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size_train, #args.batch_size,
                                              shuffle=True,
                                              pin_memory=cuda
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             pin_memory=cuda
                                             )
    print("Num classes (k) = ", len(classes), flush=True)
    return trainloader, testloader, classes, c


def get_birds(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
            ]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes

    # for CUB
    for i in range(len(classes)):
        classes[i]=classes[i].split('.')[1]

    return trainset, projectset, testset, classes, shape


def get_pets(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224):
    # COPY from get_birds
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
            ]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes

    return trainset, projectset, testset, classes, shape


def get_mura(augment: bool, train_dir: str, project_dir: str, test_dir:str, img_size = 224):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #resize to 256x256
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.4,0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15,shear=(-2, 2)),
            ]),
            transforms.RandomCrop(size=(img_size, img_size)), #crop to 224x224
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes
    
    return trainset, projectset, testset, classes, shape


def get_cars(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size=224):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size + 32, img_size + 32)),  # resize to 256x256
            transforms.RandomOrder([
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.4, 0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, shear=(-2, 2)),
            ]),
            transforms.RandomCrop(size=(img_size, img_size)),  # crop to 224x224
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes

    return trainset, projectset, testset, classes, shape


if __name__=='__main__':
    from util.args import get_args
    import pickle
    args = get_args()
    args.dataset ='MURA'
    _, _, _, classes, _ = get_data(args)
    dic = {i: k for i, k in enumerate(classes)}
    print(dic)
    with open('index2label_%s.pkl' % args.dataset , 'wb') as f:
        pickle.dump(dic, f)

    # with open('index2label_%s.pkl' % args.dataset , 'rb') as f:
    #     pickle.load(dic, f)



#########################
# def create_datasets(transform1, transform2, transform_no_augment, num_channels: int, train_dir: str, project_dir: str,
#                     test_dir: str, seed: int, validation_size: float, train_dir_pretrain=None, test_dir_projection=None,
#                     transform1p=None):
#     print(train_dir)
#     trainvalset = torchvision.datasets.ImageFolder(train_dir)
#     classes = trainvalset.classes
#     targets = trainvalset.targets
#     indices = list(range(len(trainvalset)))
#
#     train_indices = indices
#
#     if test_dir is None:
#         if validation_size <= 0.:
#             raise ValueError(
#                 "There is no test set directory, so validation size should be > 0 such that training set can be split.")
#         subset_targets = list(np.array(targets)[train_indices])
#         train_indices, test_indices = train_test_split(train_indices, test_size=validation_size,
#                                                        stratify=subset_targets, random_state=seed)
#         testset = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment),
#                                           indices=test_indices)
#         print("Samples in trainset:", len(indices), "of which", len(train_indices), "for training and ",
#               len(test_indices), "for testing.", flush=True)
#     else:
#         testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
#
#     trainset = torch.utils.data.Subset(
#         TwoAugSupervisedDataset(trainvalset, transform1=transform1, transform2=transform2), indices=train_indices)
#     trainset_normal = torch.utils.data.Subset(
#         torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=train_indices)
#     trainset_normal_augment = torch.utils.data.Subset(
#         torchvision.datasets.ImageFolder(train_dir, transform=transforms.Compose([transform1, transform2])),
#         indices=train_indices)
#     projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
#
#     if test_dir_projection is not None:
#         testset_projection = torchvision.datasets.ImageFolder(test_dir_projection, transform=transform_no_augment)
#     else:
#         testset_projection = testset
#     if train_dir_pretrain is not None:
#         trainvalset_pr = torchvision.datasets.ImageFolder(train_dir_pretrain)
#         targets_pr = trainvalset_pr.targets
#         indices_pr = list(range(len(trainvalset_pr)))
#         train_indices_pr = indices_pr
#         if test_dir is None:
#             subset_targets_pr = list(np.array(targets_pr)[indices_pr])
#             train_indices_pr, test_indices_pr = train_test_split(indices_pr, test_size=validation_size,
#                                                                  stratify=subset_targets_pr, random_state=seed)
#
#         trainset_pretraining = torch.utils.data.Subset(
#             TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1p, transform2=transform2),
#             indices=train_indices_pr)
#     else:
#         trainset_pretraining = None
#
#     return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(
#         targets)




# def get_pets(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size: int, seed: int,
#              validation_size: float):
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     normalize = transforms.Normalize(mean=mean, std=std)
#     transform_no_augment = transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#         normalize
#     ])
#
#     if augment:
#         transform1 = transforms.Compose([
#             transforms.Resize(size=(img_size + 48, img_size + 48)),
#             TrivialAugmentWideNoColor(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomResizedCrop(img_size + 8, scale=(0.95, 1.))
#         ])
#
#         transform2 = transforms.Compose([
#             TrivialAugmentWideNoShape(),
#             transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
#             transforms.ToTensor(),
#             normalize
#         ])
#     else:
#         transform1 = transform_no_augment
#         transform2 = transform_no_augment
#
#     return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed,
#                            validation_size)

#
# class TwoAugSupervisedDataset(torch.utils.data.Dataset):
#     r"""Returns two augmentation and no labels."""
#
#     def __init__(self, dataset, transform1, transform2):
#         self.dataset = dataset
#         self.classes = dataset.classes
#         if type(dataset) == torchvision.datasets.folder.ImageFolder:
#             self.imgs = dataset.imgs
#             self.targets = dataset.targets
#         else:
#             self.targets = dataset._labels
#             self.imgs = list(zip(dataset._image_files, dataset._labels))
#         self.transform1 = transform1
#         self.transform2 = transform2
#
#     def __getitem__(self, index):
#         image, target = self.dataset[index]
#         image = self.transform1(image)
#         return self.transform2(image), self.transform2(image), target
#
#     def __len__(self):
#         return len(self.dataset)
#
#
# # function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
# class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
#     def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
#         return {
#             "Identity": (torch.tensor(0.0), False),
#             "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
#             "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
#             "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
#             "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
#             "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
#         }
#
#
# class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
#     def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
#         return {
#             "Identity": (torch.tensor(0.0), False),
#             "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
#             "Color": (torch.linspace(0.0, 0.5, num_bins), True),
#             "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
#             "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
#             "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
#             "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
#             "AutoContrast": (torch.tensor(0.0), False),
#             "Equalize": (torch.tensor(0.0), False),
#         }
#
#
# class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
#     def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
#         return {
#
#             "Identity": (torch.tensor(0.0), False),
#             "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
#             "Color": (torch.linspace(0.0, 0.02, num_bins), True),
#             "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
#             "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
#             "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
#             "AutoContrast": (torch.tensor(0.0), False),
#             "Equalize": (torch.tensor(0.0), False),
#         }
