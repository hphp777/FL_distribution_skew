import multiprocessing
from multiprocessing import Process, Queue
from statistics import mode
import time
import torch
import numpy
from model.resnet import ResNet50
from model.efficientnet import EfficientNet
from train_client import client

client_num = 1

# Create client
client0 = client(0, 'resnet')
client1 = client(1, 'resnet')
client2 = client(2, 'resnet')
client3 = client(3, 'resnet')
client4 = client(4, 'resnet')

client5 = client(5, 'efficientnetb0')
client6 = client(6, 'efficientnetb0')
client7 = client(7, 'efficientnetb0')
client8 = client(8, 'efficientnetb0')
client9 = client(9, 'efficientnetb0')

clients = [client0, client1, client2, client3, client4, client5, client6, client7, client8, client9]



# Add all data number
total_data_num = 0.0

for i in range(client_num):
    total_data_num += len(clients[i].dataloader)

def centralized_server(pool):

    training_round = 2
    weights = [0] * 10
    procs = []

    
    
    # Initial Round
    for i in range(client_num):
        q = Queue()
        p = Process(target = clients[i].train, args=(q, False,))
        procs.append(p)
        p.start()
        weights[i] = q.get()
        p.join()

    for i in range (client_num):
        for key in weights[i]:
            weight = (len(clients[i].dataloader) / total_data_num) * weights[i][key]

    # weight = (len(client0.dataloader) / total_data_num) * weights[0] \
        #  + (len(client1.dataloader) / total_data_num) * weights[1] \
        #  + (len(client2.dataloader) / total_data_num) * weights[2] + (len(client3.dataloader) / total_data_num) * weights[3] \
        #  + (len(client4.dataloader) / total_data_num) * weights[4] \
        #  + (len(client5.dataloader) / total_data_num) * weights[5] \
        #  + (len(client6.dataloader) / total_data_num) * weights[6] + (len(client7.dataloader) / total_data_num) * weights[7] \
        #  + (len(client8.dataloader) / total_data_num) * weights[8] + (len(client9.dataloader) / total_data_num) * weights[9]
     
    for i in range(training_round):
        print("training round : ", i)
        for j in range(client_num):
            q = Queue()
            p = Process(target = clients[j].train, args=(q, True, weight))
            p.start()
            weights[j] = q.get()
            p.join()

        for j in range (client_num):
            for key in weights[j]:
                weight = (len(clients[j].dataloader) / total_data_num) * weights[j][key]

def peer_to_peer(): # 1:1로 weight를 교환할 때 weight값에 어떤 가중치를 줘야 하는지 잘 모르겠다. 
    pass

if __name__ == '__main__':

    pool = multiprocessing.Pool(5)

    centralized_server(pool)

    # print(client7.test())


