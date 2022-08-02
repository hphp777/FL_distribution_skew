from multiprocessing.connection import Client
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from pipelining import cifar100Loader, ChestXLoader, cifar10Loader, ChestXLoaderTest
from torch.utils.data import DataLoader
from model.resnet import ResNet50
from model.efficientnet import EfficientNet
from typing import OrderedDict

class server():

    def __init__(self, model):
        self.model_name = model
        self.batch = 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataloader = DataLoader(ChestXLoaderTest(), batch_size = self.batch, shuffle=True)
        
        # model
        if self.model_name == 'resnet':
            self.model = ResNet50.resnet56()
        elif self.model_name == 'efficientnetb0':
            self.model = EfficientNet.efficientnet_b0()

    def merge_weight(self, weights, client_num, clients, total_data_num):
        
        weight = OrderedDict()

        for i in range (client_num):
            if i == 0:
                for key in weights[i]:
                    weight[key] = (len(clients[i].dataloader) / total_data_num) * weights[i][key]
            else:
                for key in weights[i]:
                    weight[key] += (len(clients[i].dataloader) / total_data_num) * weights[i][key]

        return weight

    def test(self, weights, client_num, clients, total_data_num, round):
        
        weight = self.merge_weight(weights, client_num, clients, total_data_num)
        self.model.load_state_dict(weight)
        torch.save(self.model.state_dict(), './model/resnet/weight/global_model_round' + str(round) + 'pth')
        self.model.eval()
        dataloader = DataLoader(ChestXLoaderTest(), batch_size = self.batch,shuffle=True)

        with torch.no_grad(): # for the evaluation mode
            
            self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
            test_correct = 0.0 
            total = 0.0
            
            for _, (imgs, labels) in enumerate(tqdm(dataloader, desc= "Test Round")):

                imgs = imgs.to(self.device) # allocate data to the device
                labels = labels.cpu().detach().numpy()
                pred = self.model(imgs)
                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
                correct = (pred == labels).sum()
                test_correct += correct
                total += len(labels)
            acc = test_correct / total

        print("Global Test Accuracy : " , (test_correct / total) * 100)

        return acc

class client():
    def __init__(self, client_number , model):
        
        # hyperparameter
        self.epochs = 2 # local epochs
        self.learning_rate = 0.01
        self.weight_decay = 0.0001
        self.width_range = [0.25, 1.0]
        self.mu = 0.45
        self.cnum = client_number
        self.batch = 4
        self.model_name = model

        # model
        if self.model_name == 'resnet':
            self.model = ResNet50.resnet56()
        elif self.model_name == 'efficientnetb0':
            self.model = EfficientNet.efficientnet_b0()

        # train option
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)

        self.dataloader = DataLoader(ChestXLoader(self.cnum), batch_size = self.batch, shuffle=True)
        
    def train(self,q = None,updated = False, weight = None):
        
        self.updated = updated
        self.model.to(self.device) # allocate model to device
        self.model.train() # set model as train mode
        epoch_loss = []
        epoch_acc = []
        
        PATH = "./model.pt"
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        # self.model.load_state_dict(torch.load(PATH))

        # If I recieve the weight
        if self.updated == True:
            self.model.load_state_dict(weight)

        print("-----client" + str(self.cnum) + "-----")
        for epoch in range(self.epochs):
            
            batch_loss = []
            total_correct = 0.0
            total_data = 0.0
            self.model.train()
            # print("client " + str(self.cnum) + " training")

            for batch_idx, (imgs, labels) in enumerate(tqdm(self.dataloader, desc="Training Epoch " + str(epoch+1) + "/" + str(self.epochs))):

                imgs = imgs.to(self.device) # allocate data to the device
                labels = labels.clone().detach().type(torch.LongTensor).to(self.device)


                self.optimizer.zero_grad() # optimize the training process
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1])) # width_range = [0.25, 1.0]

                if self.model_name == 'resnet':
                    teacher_feats, teacher_output = self.model.extract_feature(imgs) # [B x 32 x 16 x 16, B x 64 x 8 x 8], B x num_classes
                elif self.model_name == 'efficientnetb0':
                    teacher_output = self.model(imgs)

                # calculate accuracy
                teacher_output_n = np.argmax(teacher_output.cpu().detach().numpy(), axis=1)
                correct = (teacher_output_n == labels.cpu().detach().numpy()).sum()
                total_correct += correct
                total_data += labels.size(0)
                loss = self.criterion(teacher_output, labels) # how to make labels??
                grad_scaler.scale(loss).backward()


                if self.model_name == 'resnet':
                    loss_CE = loss.item()
                    self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[0]))
                    s_feats = self.model.reuse_feature(teacher_feats[-2].detach())
                    # Lipschitz loss
                    TM_s = torch.bmm(self.transmitting_matrix(s_feats[-2], s_feats[-1]), self.transmitting_matrix(s_feats[-2], s_feats[-1]).transpose(2,1))
                    TM_t = torch.bmm(self.transmitting_matrix(teacher_feats[-2].detach(), teacher_feats[-1].detach()), self.transmitting_matrix(teacher_feats[-2].detach(), teacher_feats[-1].detach()).transpose(2,1))
                    loss = F.mse_loss(self.top_eigenvalue(K=TM_s), self.top_eigenvalue(K=TM_t))
                    loss = self.mu*(loss_CE/loss.item())*loss
                    grad_scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                # self.optimizer.step()
                batch_loss.append(loss.item())
            
            acc = (total_correct / total_data) * 100
            print("Train Accuracy: ", acc)

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                epoch_acc.append(acc)

        plt.plot(range(self.epochs), epoch_acc)
        plt.savefig('./result/Client' + str(self.cnum) + '_training_accuracy.png')
        plt.plot(range(self.epochs), epoch_loss)
        plt.savefig('./result/Client' + str(self.cnum) + '_training_loss.png')
        
        weights = self.model.cpu().state_dict()
        q.put(weights) # return weights as a result of the training

    def transmitting_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp

    def top_eigenvalue(self, K, n_power_iterations=10, dim=1):
        v = torch.ones(K.shape[0], K.shape[1], 1).to(self.device)
        for _ in range(n_power_iterations):
            m = torch.bmm(K, v)
            n = torch.norm(m, dim=1).unsqueeze(1)
            v = m / n

        top_eigenvalue = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
        return top_eigenvalue
    
    def test(self):

        self.model.eval()
        dataloader = DataLoader(ChestXLoaderTest(), batch_size = self.batch,shuffle=True)

        with torch.no_grad(): # for the evaluation mode
            
            self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
            test_correct = 0.0 
            total = 0.0
            
            for _, (imgs, labels) in enumerate(tqdm(dataloader, desc= "Test Round")):

                imgs = imgs.to(self.device) # allocate data to the device
                labels = labels.cpu().detach().numpy()
                pred = self.model(imgs)
                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
                correct = (pred == labels).sum()
                test_correct += correct
                total += len(labels)

        print("Test Accuracy : " , (test_correct / total) * 100)