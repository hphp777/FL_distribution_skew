import multiprocessing
from multiprocessing import Process
import time
import torch
import numpy
from pipelining import Loader
from torch.utils.data import DataLoader
from model.resnet import ResNet50
from train_client import client

batch_size = 1
local_epochs = 10

# Define each client's dataloader
dataloader0 = DataLoader(Loader(0))
dataloader1 = DataLoader(Loader(1))
dataloader2 = DataLoader(Loader(2))
dataloader3 = DataLoader(Loader(3))
dataloader4 = DataLoader(Loader(4))
dataloader5 = DataLoader(Loader(5))
dataloader6 = DataLoader(Loader(6))
dataloader7 = DataLoader(Loader(7))
dataloader8 = DataLoader(Loader(8))
dataloader9 = DataLoader(Loader(9))

# Define client model
resnet = ResNet50.resnet56()

# Create client
client0 = client(dataloader0, resnet, local_epochs)
client1 = client(dataloader1, resnet, local_epochs)
client2 = client(dataloader2, resnet, local_epochs)
client3 = client(dataloader3, resnet, local_epochs)
client4 = client(dataloader4, resnet, local_epochs)
client5 = client(dataloader5, resnet, local_epochs)
client6 = client(dataloader6, resnet, local_epochs)
client7 = client(dataloader7, resnet, local_epochs)
client8 = client(dataloader8, resnet, local_epochs)
client9 = client(dataloader9, resnet, local_epochs)

print(torch.cuda.is_available())

process0 = Process(client0.train())
weight0 = process0.start()
print(weight0)
process0.join()

# Assign train function to each process
# process_list = range(10)
# pool = multiprocessing.Pool(processes=10)
# pool.map(, process_list)
# pool.close()
# pool.join()

# start_time = time.time()
# Measure execution time : time.time() - start_time

