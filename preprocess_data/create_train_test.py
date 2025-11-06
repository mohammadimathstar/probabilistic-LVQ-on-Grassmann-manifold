import os
import shutil
import numpy as np
import time
from PIL import Image


path = './data/PETS/'

path_images = os.path.join(path,'images/')
path_train = os.path.join(path,'annotations/trainval.txt')
path_test = os.path.join(path, 'annotations/test.txt')

train_save_path = os.path.join(path,'dataset/train/')
test_save_path = os.path.join(path,'dataset/test/')
bbox_path = os.path.join(path, 'bounding_boxes.txt')

images = []
ntrain = ntest = 0
with open(path_train, 'r') as f:
    for line in f:
        file_name = line.split()[0] + '.jpg'
        pet_type = "_".join(file_name.split("_")[:-1])
        if not os.path.isdir(train_save_path + pet_type):
            os.makedirs(os.path.join(train_save_path, pet_type))

        img = Image.open(os.path.join(path_images, file_name)).convert('RGB')
        img.save(os.path.join(train_save_path + pet_type, file_name))
        print(os.path.join(path_images, file_name))

        ntrain += 1

with open(path_test, 'r') as f:
    for line in f:
        file_name = line.split()[0] + '.jpg'
        pet_type = "_".join(file_name.split("_")[:-1])
        if not os.path.isdir(test_save_path + pet_type):
            os.makedirs(os.path.join(test_save_path, pet_type))

        img = Image.open(os.path.join(path_images, file_name)).convert('RGB')
        img.save(os.path.join(test_save_path + pet_type, file_name))
        print(os.path.join(path_images, file_name))

        ntest += 1

print(f"There are {ntrain} training and {ntest} testing examples.")