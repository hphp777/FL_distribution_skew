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

class cifar100Loader(Dataset):

    def create_label(self, img_path):
        classes = [x for x in glob('C:/Users/hb/Desktop/data/CIFAR100/train/*')]
        class_names = []
        for i in range(100):
            cc = classes[i].split("\\")
            class_names.append(cc[1])

        class_name = img_path.split('\\')[1].split('_')[0]
        # print(class_name)
        # print(class_names)

        for i in range(100):
            if class_name == class_names[i]:
                # print(i)
                return i
    
    def cifar100(self, img_path):

        img = cv2.imread(img_path)
        label = self.create_label(img_path)

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        ])

        img = transform(img)

        return img, label

    def __init__(self,client_index, mode):

        self.mode = mode

        if self.mode == 'train':
            self.all_image_paths = glob('C:/Users/hb/Desktop/Data/CIFAR100_Client/C' + str(client_index) + '/*.png')
        if self.mode == 'test':
            self.all_image_paths = glob('C:/Users/hb/Desktop/Data/CIFAR100/test/*.png')
        
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

        img = cv2.imread(img_path)
        label = self.create_label(img_path)

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        ])

        img = transform(img)

        return img, label

    def __init__(self,client_index = None, mode = 'train'):
        if mode == 'train':
            self.all_image_paths = glob('C:/Users/hb/Desktop/Data/CIFAR10_Client/C' + str(client_index) + '/*.png')
        if mode == 'test':
            self.all_image_paths = glob('C:/Users/hb/Desktop/Data/CIFAR10/test/*.png')

    def __len__(self):
        return len(self.all_image_paths)
    
    def __getitem__(self, index):
        img, label = self.cifar10(self.all_image_paths[index])
        return img, label

class ChestXLoader(Dataset):
    
    def ChestXdataloader(self, img_path):
        
        resize = 128

        img = cv2.imread(img_path, IMREAD_GRAYSCALE)
        label = self.create_label(img_path)

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize,resize], PIL.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        ])
        
        img = transform(img)

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
        
        image = self.ChestXdataloader(self.all_image_paths[index])

        return image

    def create_label(self, img_path):
        if "Atelectasis" in img_path:
            return 0
        elif "Cardiomegaly" in img_path:
            return 1
        elif "Consolidation" in img_path:
            return 2
        elif "Edema" in img_path:
            return 3
        elif "Effusion" in img_path:
            return 4
        elif "Emphysema" in img_path:
            return 5
        elif "Fibrosis" in img_path:
            return 6
        elif "Hernia" in img_path:
            return 7
        elif "Infiltration" in img_path:
            return 8
        elif "Mass" in img_path:
            return 9
        elif "Nodule" in img_path:
            return 10
        elif "NoFinding" in img_path:
            return 11
        elif "Pleural_Thickening" in img_path:
            return 12
        elif "Pneumonia" in img_path:
            return 13
        elif "Pneumothorax" in img_path:
            return 14
