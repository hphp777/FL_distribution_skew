import multiprocessing
from multiprocessing import Process
from statistics import mode
import time
import torch
import numpy
from pipelining import Loader
from torch.utils.data import DataLoader
from model.resnet import ResNet50
from model.efficientnet import EfficientNet
from train_client import client

# Create client
# client0 = client(0, resnet)
# client1 = client(1, resnet)
# client2 = client(2, resnet)
# client3 = client(3, resnet)
# client4 = client(4, resnet)
# client5 = client(5, resnet)
# client6 = client(6, resnet)
client7 = client(7, 'resnet')
# client8 = client(8, resnet)
# client9 = client(9, resnet)

if __name__ == '__main__':

    train_process7 = Process(target = client7.train)
    weight7 = train_process7.start()
    train_process7.join()

    # print(client7.test())


