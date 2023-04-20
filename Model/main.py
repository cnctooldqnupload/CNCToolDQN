import os
os.environ.setdefault("TF_NUM_THREADS", "1")
import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import math
import time
import json
import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from model import *
from Enviroment import *
from scoring_function import scoring_function
from scoring import scoring_train_data
from data_open import data_open
from inference import inference

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default = '/home/alan0410/NAS_folder/Data/data/github/')
    parser.add_argument('--dir_savefigure', type = str, default = '/home/alan0410/NAS_folder/Github업로드용/save')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--maxlen", type=int, default=300)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--tau", type=float, default=0.1)
    
    args = parser.parse_args()
    
    SAVE_DIRECTORY = args.dir
    DATA_SOURCE_PATH = SAVE_DIRECTORY  #+ '/model_inputs/'  
    if 'Model' not in os.listdir(SAVE_DIRECTORY):
        os.mkdir(SAVE_DIRECTORY + '/Model')
    MODEL_STORAGE_PATH = SAVE_DIRECTORY + 'Model/q_target.pt'
    GPU_NUM = args.gpu
    max_len = args.maxlen
    
    anomaly_train_index_final, unknown_train_index, spindleload_train_data, servoload_x_train_data, servoload_y_train_data, servoload_z_train_data, life_train_data, train_label_evaluation, anomaly_train_data ,batch_num_train_data, anomaly_index_final_test, unknown_test_index, spindleload_test_data,servoload_x_test_data,servoload_y_test_data,servoload_z_test_data,life_test_data,test_label_evaluation,batch_num_test_data = data_open(DATA_SOURCE_PATH)
    
    #print(spindleload_test_data.shape, spindleload_train_data.shape)
    #print(spindleload_test_data == spindleload_train_data)
    #print(life_test_data == life_train_data)
    
    device = torch.device( f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")
    
    torch.set_num_threads(4)

    score_list = []
    env = My_policy(max_len, anomaly_train_index_final, unknown_train_index, spindleload_train_data,servoload_x_train_data,servoload_y_train_data,servoload_z_train_data ,life_train_data)
    q= Model(device,max_len)#.to(device)
    q_target = Model(device,max_len)#.to(device)
    q_target.load_state_dict(q.state_dict()) 
    memory = ReplayBuffer(device)
    print_interval = 10
    score = 0.0

    for n_epi in range(6):
        torch.set_num_threads(4)

        start = time.time()
        epsilon = max(0.01, 0.98 - 0.02 * (n_epi))
        sp_u, x_u, y_u, z_u, life_u, sp_a, x_a, y_a, z_a, life_a  = env.reset(spindleload_train_data,servoload_x_train_data,servoload_y_train_data,servoload_z_train_data ,life_train_data)
        done = False

        # Initialization       
        
        #unknown_index
        
        now_index = np.random.randint(len([unknown_train_index]))
        
        s1 = spindleload_train_data[unknown_train_index][now_index]
        s2 = servoload_x_train_data[unknown_train_index][now_index]
        s3 = servoload_y_train_data[unknown_train_index][now_index]
        s4 = servoload_z_train_data[unknown_train_index][now_index]
        
        s_life = [life_train_data[unknown_train_index][now_index]]
        
        s1, s2, s3, s4, = torch.tensor([s1]), torch.tensor([s2]), torch.tensor([s3]),  torch.tensor([s4])

        anomaly_label = anomaly_train_data[unknown_train_index][now_index]
        
        step = 0
        r_list = []
        r_extrinsic_list, r_intrinsic_list = [], []
        
        while not done:
            a, action_prob = q.sample_action(s1,s2,s3, s4, s_life, epsilon)    
            
            x_rep, out = q.forward(s1,s2, s3,s4, s_life)

            s = x_rep

            sp_prime, x_prime, y_prime,  z_prime,  s_prime, life_prime, r, r_extrinsic, r_intrinsic, done, anomaly_label_shat= env.step(q, a, action_prob, s, s1, s2, s3, s4,  s_life, anomaly_label)

            done_mask = 0.0 if done else 1.0

            r_list.append(round(r,2))
            memory.put((s[0].tolist(), s1[0].tolist(), s2[0].tolist(), s3[0].tolist(),s4[0].tolist(), s_life, a, r/5 , s_prime.tolist() , sp_prime.tolist(), x_prime.tolist(), y_prime.tolist(),  z_prime.tolist(), life_prime, done_mask, anomaly_label, anomaly_label_shat  ))

            s = s_prime
            s1 = torch.tensor([sp_prime])
            s2 = torch.tensor([x_prime])
            s3 = torch.tensor([y_prime])
            s4 = torch.tensor([z_prime])
            s_life = life_prime
            score += r
            anomaly_label = anomaly_label_shat
            #pid = pid_hat
            
            step += 1

            if done :

                score_list.append(score/step)
                step = 0
                print("n_episode: {}, score : {:.1f}, n_buffer {}, eps : {:.1f}% | - 2 ratio {:.1f}% | -1 ratio {:.1f}%".format(n_epi,
                                                    score, memory.size(), epsilon * 100,  r_list.count(-2)/len(r_list)*100, r_list.count(-1)/len(r_list)*100  ))
                
                score = 0
                
                if n_epi == 0:
                    print("time spent to perform an episode :", time.time() - start)
                #if n_epi % 5 ==0:
                #    print("r list: ", r_list)
                else:
                    pass
                break


        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
        if n_epi % 5 == 0 and n_epi != 0:    
            best_threshold = scoring_train_data(q_target, spindleload_train_data,servoload_x_train_data,servoload_y_train_data,servoload_z_train_data ,life_train_data ,  train_label_evaluation)

    torch.save(q_target.state_dict(), MODEL_STORAGE_PATH)  
    model = Model(device, max_len)
    model.load_state_dict(torch.load(MODEL_STORAGE_PATH))

    
    ## Training set
    
    result_list_flat_train = inference(spindleload_train_data, servoload_x_train_data,  servoload_y_train_data, servoload_z_train_data, life_train_data, model)
    
    #Test set
    
    result_list_flat_test = inference(spindleload_test_data, servoload_x_test_data,  servoload_y_test_data, servoload_z_test_data, life_test_data, model)
    
    
    plt.rc('font', size=20)        # font
    plt.rc('axes', labelsize=25)   # x,y axis label font
    plt.rc('xtick', labelsize=20)  # xtick font
    plt.rc('ytick', labelsize=20)  # ytick font
    plt.rc('legend', fontsize=20)  # Legend font
    plt.rc('figure', titlesize=75) 

    scaler = MinMaxScaler( feature_range=(0, 1))
    scaler.fit(np.array(result_list_flat_train))

    result_list_flat_train_scaled = scaler.transform(result_list_flat_train)   
    result_list_flat_train_scaled = 1- result_list_flat_train_scaled
    
    
    #result_list_flat_train_scaled = 1 - result_list_flat_train_scaled
    life_train_data = 1- life_train_data
    
    # Q(s, a0) plot (Training set)
    plt.figure(figsize= (20,10))
    plt.scatter(np.arange(len(result_list_flat_train )),  np.array(result_list_flat_train_scaled), c= 'r', label = 'Prediction',linewidth=2, s= 1.5)
    plt.plot(np.arange(len(life_train_data )), np.array(life_train_data ), c= 'b', label = 'Life', linewidth=3)
    #plt.legend()
    plt.title("CNCToolDQN Q(s, a0 )  (Training set)")
    plt.ylim(0, 1)
    plt.xlim(0, len(life_train_data ))
    plt.xlabel("Time (cycles)")
    plt.grid()
    plt.ylabel("Prediction")
    plt.savefig(args.dir_savefigure+ '/Q(s,a0).png')
    plt.show()
    
    RUL_list_train, mean_life_list_train ,mean_list, baseline_eliminated,  score_list = scoring_function (result_list_flat_train_scaled, life_train_data,  batch_num_train_data , args.L, args.alpha, args.tau)#L, alpha , tau
    
    # RUL plot (Training set)
    plt.figure(figsize= (20,10))
    plt.plot(np.arange(len(RUL_list_train)),RUL_list_train, c= 'r', label = 'Prediction', linewidth = 3)
    plt.plot(np.arange(len(mean_life_list_train )), np.array(mean_life_list_train ) , c= 'b', label = 'Life', linewidth=3)
    plt.grid()
    plt.title("TH score (Training set)")
    plt.xlabel("Time (batch)")
    plt.ylim(0, 1)
    plt.xlim(0, )
    plt.ylabel("TH score")
    plt.savefig(args.dir_savefigure + '/TH_score.png')
    plt.show()
    
   
    result_list_flat_test_scaled = scaler.transform(result_list_flat_test)
    
    result_list_flat_test_scaled  = 1 - result_list_flat_test_scaled 
    life_test_data = 1- life_test_data

    # Q(s, a0) plot (Test set)
    plt.figure(figsize= (20,10))
    plt.scatter(np.arange(len(result_list_flat_test )),  np.array(result_list_flat_test_scaled), c= 'r', label = 'Prediction',linewidth=2, s= 1.5)
    plt.plot(np.arange(len(life_test_data )), np.array(life_test_data), c= 'b', label = 'Life', linewidth=3)
    #plt.legend()
    plt.title("CNCToolDQN Q(s, a0 ) (Test set)")
    #plt.ylim(0, 1)
    plt.xlim(0, len(life_test_data ))
    plt.xlabel("Time (cycles)")
    plt.grid()
    plt.ylabel("Prediction")
    plt.savefig(args.dir_savefigure+ '/Q(s,a0)_test.png')
    plt.show()
    
    RUL_list_test, mean_life_list_test ,mean_list, baseline_eliminated,  score_list = scoring_function (result_list_flat_test_scaled, life_test_data, batch_num_test_data,  args.L, args.alpha, args.tau) #L, alpha , tau
    
    # RUL plot (test set)
    plt.figure(figsize= (20,10))
    plt.plot(np.arange(len(RUL_list_test)),RUL_list_test, c= 'r', label = 'Prediction', linewidth = 3)
    plt.plot(np.arange(len(mean_life_list_test )), np.array(mean_life_list_test ) , c= 'b', label = 'Life', linewidth=3)
    plt.grid()
    plt.title("TH score (Test set)")
    plt.xlabel("Time (batch)")
    #plt.ylim(0, 1)
    plt.xlim(0, )
    plt.ylabel("TH score")
    plt.savefig(args.dir_savefigure + '/TH_score_test.png')
    plt.show()
    
    
    #Printing the f1 score of test_sset
    
    from sklearn.metrics import f1_score, confusion_matrix

    r = np.where(np.array(result_list_flat_test_scaled ) > best_threshold, 1 , 0)

    f1_score = round(f1_score(test_label_evaluation, r, average ='macro'), 3)

    print("F1 score:{} ".format(f1_score))
    print("confusion:\n", confusion_matrix(test_label_evaluation,r))
