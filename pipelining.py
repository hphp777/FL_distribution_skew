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
    
    def cifar100(self, index):

        resize = 256
        if self.mode == 'train':
            img = self.client_train[index]
            label = self.client_train_label[index][0]
        elif self.mode == 'test':
            img = self.client_test[index]
            label = self.client_test_label[index][0]

        plt.imshow(img)
        plt.show()

        transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize([resize,resize], PIL.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std, inplace=False),
        ])

        img = transform(img)

        return img, label

    def __init__(self,client_index, mode):

        self.mode = mode
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar100.load_data()

        # print(self.x_train.shape)

        # Distribute data to the clients
        if self.mode == 'train':
            self.client_train = self.x_train
            self.client_train_label = self.y_train
        elif self.mode =='test':
            self.client_test = self.x_test[:2000]
            self.client_test_label = self.y_test[:2000]
        
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.client_train)
        elif self.mode == 'test':
            return  len(self.client_test)
    
    def __getitem__(self, index):
        img, label = self.cifar100(index)
        return img, label

class cifar10Loader(Dataset):
    
    def cifar10(self, index):

        if self.mode == 'train':
            img = self.client_train[index]
            label = self.client_train_label[index][0]
        elif self.mode == 'test':
            img = self.server_test[index]
            label = self.server_test_label[index][0]

        transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize([resize,resize], PIL.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std, inplace=False),
        ])

        img = transform(img)

        return img, label

    def __init__(self,client_index = None, mode = 'train'):

        self.mode = mode
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

        # Distribute data to the clients
        if self.mode == 'train':
            if client_index == 0:
                self.client_train = self.x_train[:10000]
                self.client_train_label = self.y_train[:10000]
            elif client_index == 1:
                self.client_train = self.x_train[10000:20000]
                self.client_train_label = self.y_train[10000:20000]
            elif client_index == 2:
                self.client_train = self.x_train[20000:30000]
                self.client_train_label = self.y_train[20000:30000]
            elif client_index == 3:
                self.client_train = self.x_train[30000:40000]
                self.client_train_label = self.y_train[30000:40000]
            elif client_index == 4:
                self.client_train = self.x_train[40000:50000]
                self.client_train_label = self.y_train[40000:50000]
        elif self.mode =='test':
            self.server_test = self.x_test
            self.server_test_label = self.y_test
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.client_train)
        elif self.mode == 'test':
            return  len(self.server_test)
    
    def __getitem__(self, index):
        img, label = self.cifar10(index)
        return img, label

class ChestXLoaderTest(Dataset):
    def ChestXdataloader(self, img_path):
        
        resize = 256
        mean = [0.485]
        std = [0.229] 

        img = cv2.imread(img_path)
        label = self.create_label(img_path)

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize,resize], PIL.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std, inplace=False),
        ])
        
        img = transform(img)

        return img, label

    def __init__(self):
        self.all_image_paths = []
        for client_index in range(10):
            cdata = glob('C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(client_index) + '/*.png')

            total = len(cdata)
            self.partition = int(total * 0.8)

            self.all_image_paths += cdata[self.partition:]

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

class ChestXLoader(Dataset): # custom dataset
    def ChestXdataloader(self, img_path):
        
        resize = 64
        mean = [0.485]
        std = [0.229] 

        img = cv2.imread(img_path)
        label = self.create_label(img_path)

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([resize,resize], PIL.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std, inplace=False),
        ])
        
        img = transform(img)

        return img, label

    def __init__(self,client_index):
        
        self.all_image_paths = glob('C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(client_index) + '/*.png')
        # self.all_image_paths = glob('C:/Users/hb/Desktop/Data/ChestX-ray14/*/*.png')
        
        total = len(self.all_image_paths)
        self.partition = int(total * 0.8)

        self.all_image_paths = self.all_image_paths[:self.partition]

        
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