import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import PIL.Image as PIL
import torchvision.transforms as transforms
import os
from glob import glob

class Loader(Dataset): # custom dataset
    def ChestXdataloader(self, img_path):
        
        resize = 256
        mean = [0.485]
        std = [0.229] 

        img = PIL.open(img_path)

        transform = transforms.Compose([
        transforms.Resize([resize,resize], PIL.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std, inplace=False),
        ])
        
        img = transform(img)

        return img

    def __init__(self,client_index):
        self.all_image_paths = glob('C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(client_index) + '/*.png')
        
    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index): # 여기에 들어가는 index는?
        
        if torch.is_tensor(index):
            index = index.tolist()
        
        image = self.ChestXdataloader(self.all_image_paths[index])

        return image