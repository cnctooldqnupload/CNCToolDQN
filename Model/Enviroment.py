import os
os.environ.setdefault("TF_NUM_THREADS", "1")
import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from pytorch_metric_learning import losses

#anomaly_index_final = np.load('Data/model_inputs/anomaly_index_final.npy')
#unknown_train_index = np.load('Data/model_inputs/unknown_train_index.npy')
#spindleload_train_data = np.load('Data/model_inputs/spindleload_train_data.npy')
#servox_train_data = np.load('Data/model_inputs/servox_train_data.npy')
#servoy_train_data = np.load('Data/model_inputs/servoy_train_data.npy')
#servoz_train_data = np.load('Data/model_inputs/servoz_train_data.npy')
#life_train_data = np.load('Data/model_inputs/life_train_data.npy')

class My_policy():
    def __init__(self, win_size, anomaly_index_final, unknown_train_index, spindleload_train_data,servox_train_data,servoy_train_data,servoz_train_data ,life_train_data  ):
        #self.state = np.zeros(370)
        #self.state_list = [self.state]
        
        self.iforest = IsolationForest(n_estimators = 100)
        self.unknown_index = unknown_train_index
        self.anomaly_index = anomaly_index_final

        self.sp_u = spindleload_train_data[self.unknown_index]
        self.x_u = servox_train_data[self.unknown_index]
        self.y_u = servoy_train_data[self.unknown_index]
        self.z_u = servoz_train_data[self.unknown_index]
        self.life_u = life_train_data[self.unknown_index]
        
        self.sp_a = spindleload_train_data[self.anomaly_index ]
        self.x_a = servox_train_data[self.anomaly_index ]
        self.y_a = servoy_train_data[self.anomaly_index ]
        self.z_a = servoz_train_data[self.anomaly_index ]
        self.life_a = life_train_data[self.anomaly_index ]
        
        self.win_size = win_size
        
        self.search_list= []
        
    def step(self, q, action, action_prob, s, s1, s2, s3, s4, s_life, anomaly_label ): # a: anomaly_label
                
        random_idx = np.random.choice(len(self.sp_u), 2000, replace= False)

        D_u, out_u = q.forward(self.sp_u[random_idx],self.x_u[random_idx] ,self.y_u[random_idx] ,self.z_u[random_idx] , self.life_u[random_idx]) 

        D_u = D_u.detach().numpy()
        
        del out_u
        
        random_idx_anomaly = np.random.choice(len(self.sp_a), 2000, replace= False)
       
        D_a, out_a = q.forward(self.sp_a[random_idx_anomaly ],self.x_a[random_idx_anomaly ] ,self.y_a [random_idx_anomaly ], self.z_a[random_idx_anomaly ] ,self.life_a[random_idx_anomaly ])
        D_a = D_a.detach().numpy()

        del out_a        
        
        self.iforest.fit(D_u)
        predict = self.iforest.score_samples(s.detach().numpy())[0]
        
        if predict >= -1:
            reward_intrinsic = 0
            
        if predict < -1:
            reward_intrinsic = 1
                
        if action == 1 and anomaly_label == 'Anomaly':
            
            reward_extrinsic = 2
            shat, life_hat, index, anomaly_label_shat, select= self.state_movement(q, s, s1, s2, s3, s4, s_life, D_a, D_u, self.life_a, self.life_u[random_idx] ,  action, random_idx)

            self.search_list.append(shat)

            
        if action == 0 and anomaly_label == 'Anomaly' :
            
            reward_extrinsic = -2
            shat, life_hat, index, anomaly_label_shat, select  = self.state_movement(q, s, s1, s2, s3, s4,  s_life, D_a, D_u,  self.life_a, self.life_u[random_idx] ,  action, random_idx)
            
            self.search_list.append(shat)
            
        if anomaly_label == 'unknown':
                        
            if s_life[0] >= 0.5 and action_prob[1] > 0.5:
                reward_extrinsic = torch.tensor(  [min( -torch.log(torch.abs(action_prob[1] - s_life[0])), 10)] ).tolist()[0]  # Preventing reward explosion to infinity 
            
            elif s_life[0] < 0.5 and action_prob[1] < 0.5:
                reward_extrinsic = torch.tensor( [min( -torch.log(torch.abs(action_prob[1] - s_life[0] )), 10)]  ).tolist()[0] # Preventing reward explosion to infinity 

            else:
                
                reward_extrinsic = -1
                
            shat, life_hat,  index, anomaly_label_shat, select = self.state_movement(q, s, s1, s2, s3,  s4, s_life,  D_a, D_u, self.life_a, self.life_u[random_idx], action, random_idx)
            self.search_list.append(shat)     
                    
        
        if select ==  0: # Anomaly 
            sp_hat, x_hat, y_hat, z_hat = self.sp_a[index], self.x_a[index], self.y_a[index], self.z_a[index]
        
        elif select == 1: # Unknown
            sp_hat, x_hat, y_hat, z_hat = self.sp_u[index], self.x_u[index], self.y_u[index] , self.z_u[index]    
        
        reward = reward_extrinsic  + reward_intrinsic

        done = self.is_done()
        
        if done == True:
            self.search_list = []
                    
        else:
            pass

        return sp_hat, x_hat, y_hat, z_hat,  shat, [life_hat], reward, reward_extrinsic, reward_intrinsic, done, anomaly_label_shat
    
    def state_movement(self, q, s, s1, s2, s3, s4, s_life, D_a, D_u, life_a, life_u, action, random_idx):
        coin = random.random()

        if coin < 0.4:  # S_da
            index = np.random.randint(len(D_a))
            s_hat = D_a[index].reshape(1,-1)
            life_hat = life_a[index]
            #pid_hat = self.pid_a[index]

            anomaly_label = 'Anomaly'
            select = 0
            
            return s_hat, life_hat, index, anomaly_label, select
        
        else: # S_du 
                        
            s, D_u_resized = s.detach().numpy(), D_u
                                    
            euc_dist = euclidean_distances(D_u_resized, s )
                
            if action == 1:       
                index = random_idx[np.argmin(euc_dist.flatten())]

                s_hat, out_u_hat = q.forward(self.sp_u[index].reshape(1,self.win_size),self.x_u[index].reshape(1,self.win_size) ,self.y_u[index].reshape(1,self.win_size) ,self.z_u[index].reshape(1,self.win_size), torch.tensor(self.life_u[index]).reshape(1,) )

                del out_u_hat

                life_hat = self.life_u[index]
                anomaly_label = 'unknown'

            else: 
                index = random_idx[np.argmax(euc_dist.flatten())]
                
                s_hat, out_u_hat = q.forward(self.sp_u[index].reshape(1,self.win_size),self.x_u[index].reshape(1,self.win_size) ,self.y_u[index].reshape(1,self.win_size) ,self.z_u[index].reshape(1,self.win_size), torch.tensor(self.life_u[index]).reshape(1,) )
                
                del out_u_hat
                
                life_hat = self.life_u[index]
                anomaly_label = 'unknown'
            
            select = 1
            
            return s_hat, life_hat, index, anomaly_label, select
        
    def is_done(self):  # Termination 
            
        if len(self.search_list) >= 150:
            return True
        else:
            return False

        
    def reset(self, spindleload_train_data,servox_train_data,servoy_train_data,servoz_train_data ,life_train_data):
        self.sp_u = spindleload_train_data[self.unknown_index]
        self.x_u = servox_train_data[self.unknown_index]
        self.y_u = servoy_train_data[self.unknown_index]
        self.z_u = servoz_train_data[self.unknown_index]
        self.life_u = life_train_data[self.unknown_index]
        
        self.sp_a = spindleload_train_data[self.anomaly_index ]
        self.x_a = servox_train_data[self.anomaly_index ]
        self.y_a =servoy_train_data[self.anomaly_index ]
        self.z_a = servoz_train_data[self.anomaly_index ]
        self.life_a = life_train_data[self.anomaly_index ]
        
        self.search_list= []
        return self.sp_u, self.x_u, self.y_u, self.z_u, self.life_u ,self.sp_a, self.x_a, self.y_a, self.z_a, self.life_a 
