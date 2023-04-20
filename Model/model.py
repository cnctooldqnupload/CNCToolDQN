import os
os.environ.setdefault("TF_NUM_THREADS", "1")
import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from Attention_layer import Attention_layer
from pytorch_metric_learning import losses
import argparse


class ReplayBuffer():
    def __init__(self, device):

        self.buffer_limit = 7000
        self.buffer = collections.deque(maxlen = self.buffer_limit)
        self.device = device
        
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)   
        
        s_lst, s1_lst , s2_lst, s3_lst, s4_lst, s_life_lst, a_lst, r_lst, s_prime_lst,s1_prime_lst, s2_prime_lst, s3_prime_lst, s4_prime_lst, life_prime_lst, done_mask_lst =  [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        anomaly_label_lst, anomaly_label_shat_lst = [], []
        #pos_neg_label_list, pos_neg_label_hat_list = [], []
        
        for transition in mini_batch:
            s, s1, s2, s3,s4,  s_life, a, r, s_prime, s1_prime,s2_prime,s3_prime,s4_prime, life_prime, done_mask ,  anomaly_label, anomaly_label_shat = transition
            
            s_lst.append(s)
            s1_lst.append(s1)
            s2_lst.append(s2)
            s3_lst.append(s3)
            s4_lst.append(s4)
            #s5_lst.append(s5)
            
            s_life_lst.append(s_life)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            s1_prime_lst.append(s1_prime)
            s2_prime_lst.append(s2_prime)
            s3_prime_lst.append(s3_prime)
            s4_prime_lst.append(s4_prime)
            #s5_prime_lst.append(s5_prime)
            
            life_prime_lst.append(life_prime)
            done_mask_lst.append([done_mask])
            anomaly_label_lst.append(anomaly_label)
            anomaly_label_shat_lst.append(anomaly_label_shat)
            #pos_neg_label_hat_list.append(pid_hat)
            #pos_neg_label_list.append(pid)
        
        
        anomaly_label_lst = [0 if value=='Unknown' else 1 for value in anomaly_label_lst]
        #anomaly_label_lst = [0 if value=='Unknown' else 1]

    
        return torch.tensor(s_lst, dtype = torch.float).to(self.device), torch.tensor(s1_lst, dtype = torch.float).to(self.device), torch.tensor(s2_lst, dtype = torch.float).to(self.device), torch.tensor(s3_lst, dtype = torch.float).to(self.device), torch.tensor(s4_lst, dtype = torch.float).to(self.device),torch.tensor(s_life_lst, dtype = torch.float).to(self.device), torch.tensor(a_lst).to(self.device), torch.tensor(r_lst).to(self.device), torch.tensor(s_prime_lst, dtype = torch.float).to(self.device),  torch.tensor(s1_prime_lst, dtype = torch.float).to(self.device), torch.tensor(s2_prime_lst, dtype = torch.float).to(self.device), torch.tensor(s3_prime_lst, dtype = torch.float).to(self.device), torch.tensor(s4_prime_lst, dtype = torch.float).to(self.device),torch.tensor(life_prime_lst, dtype = torch.float).to(self.device), torch.tensor(done_mask_lst).to(self.device), torch.tensor(anomaly_label_lst).to(self.device)
        #s,s1, s2, s3, s4, s5, s_life, a,r,s_prime,s1_prime, s2_prime, s3_prime, s4_prime, s5_prime, life_prime, done_mask, anomaly_label, pos_neg_label, pos_neg_label_prime
    
    def size(self):
        return len(self.buffer)


class Model(nn.Module):
    def __init__(self, device, win_size):
        super(Model, self).__init__()

        self.device = device

        self.win_size = win_size ## Window size
        self.input_c = 4
        self.output_c = 1
        
        self.fc0 = nn.Linear(self.win_size, 1)
        
        self.fc1 = nn.Linear(self.win_size, 15 ).to(self.device)

        self.Flatten = nn.Flatten()
        
        self.fc2 = nn.Linear(16, 4).to(self.device)
        self.fc3 = nn.Linear(4, 2).to(self.device)
        self.batchNorm = nn.BatchNorm1d(1)
        
        self.transformer_model = Attention_layer(device = self.device, win_size=self.win_size, enc_in=self.input_c, c_out= self.output_c, e_layers= 1 ).to(self.device)  # default input c 38 output c 38
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()     
    def forward(self, x1, x2, x3, x4, x_life): # input (s, a) output Q 
        
        #print("x1.shape: ", x1.shape)
        x1 = torch.tensor(x1).float().unsqueeze(-1) #.to(self.device)
        x2 = torch.tensor(x2).float().unsqueeze(-1) #.to(self.device)
        x3 = torch.tensor(x3).float().unsqueeze(-1) #.to(self.device)
        x4 = torch.tensor(x4).float().unsqueeze(-1) #.to(self.device)
        
        x_life = 5 * torch.exp(torch.tensor(x_life)).float().unsqueeze(1) #.to(self.device)
        
        x = torch.cat([x1, x2, x3, x4], dim= -1) #.to(self.device)
                
        x_tf = self.transformer_model.forward(x).squeeze(-1)
        
        x_rep = F.relu(self.fc1(x_tf)) #.to(self.device)         
        
        del x_tf
        
        x = torch.cat([x_rep.to(self.device), x_life.to(self.device)], dim = -1)
        x= F.relu(self.fc2(x)) #.to(self.device)
        x = F.relu(self.fc3(x)) #.to(self.device)
        out = x.squeeze(1) #.to(self.device)
        out = F.softmax(out).to(self.device)

        return x_rep.squeeze(1).to('cpu'), out
    
    def sample_action(self, s1, s2, s3, s4,  s_life, epsilon):

        coin = random.random()
                
        if coin < epsilon:
            choice = random.randint(0,1)
            tensor = torch.zeros(2)
            tensor[choice] = 1
                    
            return choice, tensor
        
        else:
            x_rep, out = self.forward(s1, s2, s3,s4, s_life)
            
            return out.argmax().item(), out[0] 
        

'''
The below can be implemented when implementing contrastive loss

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks, device):
        super(MultiTaskLoss, self).__init__()
        self.tasks = nn.ModuleList(tasks)
        self.sigma1 = nn.Parameter(torch.tensor(0.5))
        self.sigma2 = nn.Parameter(torch.tensor(1.0))
        self.smooth_l1_loss = F.smooth_l1_loss
        self.device = device
        #self.NTXent_loss = losses.NTXentLoss(temperature=0.7)
        #self.contrastive_loss = losses.ContrastiveLoss(pos_margin= 0, neg_margin=1)
        
    def forward(self, s1_prime, s2_prime, s3_prime, s4_prime,  life_prime, done_mask, gamma, r ,q_a , x_res,anomaly_label):
        
        q_target = self.tasks[0]
        
        max_q_prime = q_target(s1_prime,s2_prime, s3_prime,s4_prime,   life_prime)[1].max(1)[0].to(self.device)            
        max_q_prime = max_q_prime.unsqueeze(1)
        
        target = r + gamma * max_q_prime * done_mask
        
        #print("r: ", r)
        #print("max_q_prime", max_q_prime.shape)
        #print("done mask: ", done_mask.shape)
        #print("target: ", target)
        #print("q_a", q_a)
        #r:  torch.Size([32, 1])
        #max_q_prime torch.Size([32])
        #done mask:  torch.Size([32, 1])
        #contrastive_label = pos_neg_label_prime + anomaly_label
        #x_mat = torch.cdist(x_res, x_res)

        #loss= torch.mul (1/ (self.sigma1**2), F.smooth_l1_loss(q_a, target))  + torch.mul( 1/(self.sigma2**2), self.contrastive_loss(x_res, contrastive_label)) + torch.log(self.sigma1) + torch.log(self.sigma2)
        
        loss =  F.smooth_l1_loss(q_a, target)  # + 0.2 *  self.contrastive_loss(x_res, contrastive_label)
        #print("sigma1:{}, sigma2:{}".format(self.sigma1, self.sigma2))
        return loss
'''

'''
class QAgent_Qlearning():
    def __init__(self):
        self.gamma = 0.99
        self.batch_size = 32
        self.mtl = MultiTaskLoss([q_target])
        self.learning_rate =  0.0008
        self.optimizer = optim.Adam(self.mtl.parameters(), lr = self.learning_rate)
        self.loss_list= []
    def train(self, q, memory):
        
        for i in range(10):
            
            s,s1, s2, s3, s4,  s_life, a,r,s_prime,s1_prime, s2_prime, s3_prime, s4_prime, life_prime, done_mask, anomaly_label = memory.sample(self.batch_size)
            
            s_life , life_prime = s_life.reshape(self.batch_size, ), life_prime.reshape(self.batch_size, )
            
            x_res, q_out = q(s1, s2, s3, s4, s_life) # Model 
            q_out = q_out.reshape(self.batch_size, 2)
            q_a = q_out.gather(1, a).to(self.device)             
            
            #print("action: " , a)
            #print("reward: " , r)
            
            loss = self.mtl(s1_prime,s2_prime, s3_prime, s4_prime, life_prime, done_mask, self.gamma, r ,q_a , x_res,  anomaly_label )
            #print("DQN loss :{}, constrastive loss :{}".format(F.smooth_l1_loss(q_a, target), self.contrastive_loss(x_res, anomaly_label)))
            self.loss_list.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        print("loss: ", torch.mean(torch.tensor(self.loss_list)))
'''



