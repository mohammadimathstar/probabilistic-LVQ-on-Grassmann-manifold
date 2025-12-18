
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
    if args.dataset == 'SkinCancerISIC':
        return get_cars(True,
                        './data/SkinCancer-ISIC/Train',
                        './data/SkinCancer-ISIC/Train',
                        './data/SkinCancer-ISIC/Test')
    if args.dataset == 'Chest':
        return get_cars(True,
                        './data/chest_xray/train',
                        './data/chest_xray/val',
                        './data/chest_xray/test')
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


def get_skins(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size=224):
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
            # This randomly crops and resizes the image. It's very effective.
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # We increase the rotation amount from 20 to 30 degrees.
            transforms.RandomRotation(30),
            # We make the color changes a bit stronger.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transform_no_augment # Using ImageNet stats from Part 3
        ])
        # transform = transforms.Compose([
        #     transforms.Resize(size=(img_size + 32, img_size + 32)),  # resize to 256x256
        #     transforms.RandomOrder([
        #         transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        #         transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.4, 0.4)),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomAffine(degrees=15, shear=(-2, 2)),
        #     ]),
        #     transforms.RandomCrop(size=(img_size, img_size)),  # crop to 224x224
        #     transforms.ToTensor(),
        #     normalize,
        # ])
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
    args.dataset ='SkinCancerISIC'
    _, _, _, classes, _ = get_data(args)
    dic = {i: k.replace(' ', '_') for i, k in enumerate(classes)}
    print(dic)
    with open('index2label_%s.pkl' % args.dataset , 'wb') as f:
        pickle.dump(dic, f)

    # with open('index2label_%s.pkl' % args.dataset , 'rb') as f:
    #     pickle.load(dic, f)

