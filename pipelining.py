import cv2
from cv2 import IMREAD_GRAYSCALE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import PIL.Image as PIL
import torchvision.transforms as transforms
import os
from glob import glob
import random
import pickle
import tensorflow as tf

class cifar100Loader(Dataset):
    
    def __init__(self,client_index, mode):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        
    
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass

class Loader(Dataset): # custom dataset
    def ChestXdataloader(self, img_path):
        
        resize = 256
        mean = [0.485]
        std = [0.229] 

        img = cv2.imread(img_path, IMREAD_GRAYSCALE)
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

    def __init__(self,client_index, mode):
        
        random.seed(100)
        self.all_image_paths = glob('C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(client_index) + '/*.png')
        # self.all_image_paths = glob('C:/Users/hb/Desktop/Data/ChestX-ray14/*/*.png')
        random.shuffle(self.all_image_paths)
        
        total = len(self.all_image_paths)
        self.partition = int(total * 0.8)
        self.mode = mode

        if self.mode == 'train':
            self.all_image_paths = self.all_image_paths[:self.partition]
        elif self.mode == 'test':
            self.all_image_paths = self.all_image_paths[self.partition:]
        
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