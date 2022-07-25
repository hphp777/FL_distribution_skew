from multiprocessing.connection import Client
import torch
import torch.nn.functional as F
from tqdm import tqdm

class client():
    def __init__(self, dataLoader, model, epochs):
        
        # hyperparameter
        self.dataLoader = dataLoader # dataset
        self.model = model # train archittecture
        self.epochs = epochs # epochs
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.width_range = [0.25, 1.0]
        self.mu = 0.45

        # train option
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        
    def train(self):
        
        self.model.to(self.device) # allocate model to device
        self.model.train() # set model as train mode
        epoch_loss = []

        for epoch in range(self.epochs):
            
            batch_loss = []

            for batch_idx, (imgs, labels) in enumerate(tqdm(self.dataLoader)):

                imgs = imgs.to(self.device) # allocate data to the device
                labels = torch.tensor(labels)
                self.optimizer.zero_grad() # optimize the training process
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1])) # width_range = [0.25, 1.0]
                teacher_feats, teacher_output = self.model.extract_feature(imgs) # [B x 32 x 16 x 16, B x 64 x 8 x 8], B x num_classes
                loss = self.criterion(teacher_output, labels) # how to make labels??
                loss.backward()
                loss_CE = loss.item()
                
                self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[0]))

                s_feats = self.model.reuse_feature(teacher_feats[-2].detach())

                # Lipschitz loss
                TM_s = torch.bmm(self.transmitting_matrix(s_feats[-2], s_feats[-1]), self.transmitting_matrix(s_feats[-2], s_feats[-1]).transpose(2,1))
                TM_t = torch.bmm(self.transmitting_matrix(teacher_feats[-2].detach(), teacher_feats[-1].detach()), self.transmitting_matrix(teacher_feats[-2].detach(), teacher_feats[-1].detach()).transpose(2,1))
                loss = F.mse_loss(self.top_eigenvalue(K=TM_s), self.top_eigenvalue(K=TM_t))
                loss = self.mu*(loss_CE/loss.item())*loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                
        weights = self.model.cpu().state_dict()

        return weights # return weights as a result of the training

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

        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0 
        test_sample_number = 0.0

        with torch.no_grad():
            self.model.apply(lambda m: setattr(m, 'width_mult', self.width_range[-1]))
            for batch_idx, (imgs, labels) in enumerate(self.dataLoader):
                imgs = imgs.to(self.device) # allocate data to the device
                labels = torch.tensor(labels)

                pred = self.model(imgs)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(labels).sum()
                test_correct += correct.item()
                test_sample_number += labels.size(0)

            acc = (test_correct / test_sample_number)*100
            print(acc)

        return acc

