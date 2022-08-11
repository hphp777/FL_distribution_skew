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
client5 = client(0, 'resnet')
client6 = client(1, 'resnet')
client7 = client(2, 'resnet')
client8 = client(3, 'resnet')
client9 = client(4, 'resnet')

clients = [client0, client1, client2, client3, client4, client5, client6, client7, client8, client9]

ca1 = []
ca2 = []
ca3 = []
ca4 = []
ca5 = []
ca6 = []
ca7 = []
ca8 = []
ca9 = []
ca10 = []

client_accs = [ca1,ca2,ca3,ca4,ca5, ca6, ca7, ca8 ,ca9,ca10]
clients_weight = [0] * 100

# Add all data number
total_data_num = 0.0

for i in range(client_num):
    total_data_num += len(clients[i].dataloader)

def draw_train(accs, cnum, epochs):
    plt.plot(range(epochs), accs)
    plt.savefig("./result/Training_accuracy_" + str(cnum) + ".png")
    plt.clf()

def centralized_server():

    training_round = 1
    weights = [0] * 10
    server_acc = []

    # Initial Round
    print("training round : ", 1)
    for i in range(client_num):
        acc, weights[i] = clients[i].train()
        client_accs[i].append(acc)

    weight = central_server.merge_weight(weights, client_num, clients, total_data_num)
    server_acc.append(central_server.test(weights, client_num, clients, total_data_num, 1))

    for i in range(training_round):
        print("training round : ", i+2)
        weights = [0] * 10
        for j in range(client_num):
            acc, weights[j] = clients[j].train(updated= True, weight= weight, t_r = i + 2)
            client_accs[j].append(acc)

        server_acc.append(central_server.test(weights, client_num, clients, total_data_num,i+2))

    for i in range(client_num):
        draw_train(client_accs[i], i, training_round + 1 )

    plt.plot(range(training_round + 1), server_acc)
    plt.savefig('./result/Server_test_accuracy.png')
    plt.clf()

def SOLO():

    training_round = 20
    weights = [0] * 10

    # Initial Round
    for i in range(training_round):
        print("training round : ", i+1)
        weights = [0] * 10
        for j in range(client_num):
            acc, weights[j] = clients[j].train()
            client_accs[j].append(acc)

    central_server.test(weights, client_num, clients, total_data_num,i+2)

    for i in range(client_num):
        draw_train(client_accs[i], i, training_round + 1 )

def peer_to_peer(): # 1:1로 weight를 교환할 때 weight값에 어떤 가중치를 줘야 하는지 잘 모르겠다. 
    pass

if __name__ == '__main__':

    # centralized_server()
    SOLO()



