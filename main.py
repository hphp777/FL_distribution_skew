import multiprocessing
from multiprocessing import Process, Queue
from statistics import mode
import time
from typing import OrderedDict
import torch
import numpy
from model.resnet import ResNet50
from model.efficientnet import EfficientNet
from train_client import client, server
from matplotlib import pyplot as plt

client_num = 10

# Create server
central_server = server('resnet')

# Create client
client0 = client(0, 'resnet')
client1 = client(1, 'resnet')
client2 = client(2, 'resnet')
client3 = client(3, 'resnet')
client4 = client(4, 'resnet')
client5 = client(5, 'resnet')
client6 = client(6, 'resnet')
client7 = client(7, 'resnet')
client8 = client(8, 'resnet')
client9 = client(9, 'resnet')

clients = [client0, client1, client2, client3, client4, client5, client6, client7, client8, client9]

# Add all data number
total_data_num = 0.0

for i in range(client_num):
    total_data_num += len(clients[i].dataloader)

def centralized_server():

    training_round = 2
    weights = [0] * 10
    server_acc = []

    # Initial Round
    print("training round : ", 1)
    for i in range(client_num):
        q = Queue()
        p = Process(target = clients[i].train, args=(q, False,))
        p.start()
        weights[i] = q.get()
        p.join()

    weight = central_server.merge_weight(weights, client_num, clients, total_data_num,1)
    server_acc.append(central_server.test(weights, client_num, clients, total_data_num))

    for i in range(training_round):
        print("training round : ", i+2)
        for j in range(client_num):
            q = Queue()
            p = Process(target = clients[j].train, args=(q, True, weight))
            p.start()
            weights[j] = q.get()
            p.join()

        server_acc.append(central_server.test(weights, client_num, clients, total_data_num,i+2))

    plt.plot(range(training_round + 1), server_acc)
    plt.savefig('./result/Server_test_accuracy.png')

def peer_to_peer(): # 1:1로 weight를 교환할 때 weight값에 어떤 가중치를 줘야 하는지 잘 모르겠다. 
    pass

if __name__ == '__main__':

    centralized_server()

    # print(client7.test())


