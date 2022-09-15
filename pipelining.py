from tkinter.messagebox import NO
import cv2
from cv2 import IMREAD_GRAYSCALE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import PIL.Image as PIL
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import os
from glob import glob
import random
import pickle
import tensorflow as tf
from PIL import Image, ImageOps


class cifar100Loader(Dataset):

    def create_label(self, img_path):
        classes = [x for x in glob('C:/Users/hb/Desktop/data/CIFAR100/train/*')]
        class_names = []
        for i in range(100):
            cc = classes[i].split("\\")
            class_names.append(cc[1].split('_')[0])

        class_name = img_path.split('\\')[1].split('_')[0]      

        for i in range(100):
            if class_name == class_names[i]:
                return i

    
    def cifar100(self, img_path):

        img = cv2.imread(img_path)
        label = self.create_label(img_path)

        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        ])

        train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        img = train_transform(img)

        return img, label

    def __init__(self,client_index = None, mode = None):

        self.mode = mode

        if self.mode == 'train':
            self.all_image_paths = glob('C:/Users/hb/Desktop/Data/CIFAR100_Client/C' + str(client_index) + '/*.png')
        if self.mode == 'test':
            self.all_image_paths = glob('C:/Users/hb/Desktop/Data/CIFAR100/test/*/*.png')
        
    def __len__(self):
        return(len(self.all_image_paths))
    
    def __getitem__(self, index):
        img, label = self.cifar100(self.all_image_paths[index])
        return img, label

class cifar10Loader(Dataset):
    
    def create_label(self, img_path):
        classes = [x for x in glob('C:/Users/hb/Desktop/data/CIFAR10/train/*')]
        class_names = []
        for i in range(10):
            cc = classes[i].split("\\")
            class_names.append(cc[1])

        class_name = img_path.split('\\')[1].split('_')[0]

        for i in range(10):
            if class_name == class_names[i]:
                return i

    def cifar10(self, img_path):

        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]

        img = cv2.imread(img_path)
        label = self.create_label(img_path)

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        img = transform(img)

        return img, label

    def __init__(self,client_index = None, mode = 'train'):
        if mode == 'train':
            self.all_image_paths = glob('C:/Users/hb/Desktop/Data/CIFAR10_Client_random/C' + str(client_index) + '/*.png')
        if mode == 'test':
            self.all_image_paths = glob('C:/Users/hb/Desktop/Data/CIFAR10/test/*/*.png')

    def __len__(self):
        return len(self.all_image_paths)
    
    def __getitem__(self, index):
        img, label = self.cifar10(self.all_image_paths[index])
        return img, label

class ChestXLoader(Dataset):
    
    def ChestXdataloader(self, img_path):
        
        resize = 256

        img = cv2.imread(img_path, IMREAD_GRAYSCALE)
        # img = ImageOps.grayscale(Image.open(img_path)) 
        label = self.create_label(img_path)

        # Adaptive masking
        threshold = img.min() + (img.max() - img.min()) * 0.9
        img[img > threshold] = 0

        # plt.imshow(img)
        # plt.show()

        normalize = transforms.Normalize(mean=[0.485],
                                     std=[0.229])

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize,resize], PIL.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
        ])

        img = transform(img)

        # img = img/255.0

        return img, label

    def __init__(self, client_index = None, mode = 'train'):

        if mode == 'train':
            self.all_image_paths = glob('C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(client_index) + '/*.png')
        if mode  == 'test':
            self.all_image_paths = glob('C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/test/*.png')

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index): # 여기에 들어가는 index는?
        
        if torch.is_tensor(index):
            index = index.tolist()
        
        image, label = self.ChestXdataloader(self.all_image_paths[index])

        return image, label

    def create_label(self, img_path):
        if "Atelectasis" in img_path:
            return 0
        elif "Cardiomegaly" in img_path:
            return 1
        elif "Consolidation" in img_path:
            return 2
        elif "Effusion" in img_path:
            return 3
        elif "Fibrosis" in img_path:
            return 4
        elif "Infiltration" in img_path:
            return 5
        elif "Mass" in img_path:
            return 6
        elif "Nodule" in img_path:
            return 7
        elif "Pleural_Thickening" in img_path:
            return 8
        elif "Pneumonia" in img_path:
            return 9
        elif "Pneumothorax" in img_path:
            return 10
        elif "Edema" in img_path:
            return 11
        elif "Emphysema" in img_path:
            return 12
        elif "Hernia" in img_path:
            return 13
        elif "Nofinding" in img_path:
            return 14

    def count_imbalance(self, client_index):

        self.all_image_paths = glob('C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(client_index) + '/*.png')
        label_cnt = [0] * 11

        for img_path in range(len(self.all_image_paths)):
            if "Atelectasis" in img_path:
                label_cnt[0] += 1
            elif "Cardiomegaly" in img_path:
                label_cnt[1] += 1
            elif "Consolidation" in img_path:
                label_cnt[2] += 1
            elif "Effusion" in img_path:
                label_cnt[3] += 1
            elif "Fibrosis" in img_path:
                label_cnt[4] += 1
            elif "Infiltration" in img_path:
                label_cnt[5] += 1
            elif "Mass" in img_path:
                label_cnt[6] += 1
            elif "Nodule" in img_path:
                label_cnt[7] += 1
            elif "Pleural_Thickening" in img_path:
                label_cnt[8] += 1
            elif "Pneumonia" in img_path:
                label_cnt[9] += 1
            elif "Pneumothorax" in img_path:
                label_cnt[10] += 1

        variance = np.var(label_cnt)
        
        return variance

