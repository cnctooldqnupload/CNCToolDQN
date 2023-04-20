## Scoring
import numpy as np

def scoring_function (array, life_array, batch_array , L, alpha , tau):
    
    RUL_list= []
    mean_life_list = []
    mean_list= []
    baseline_eliminated_list = []
    score_list = []
    score = 0
    
    batch_array = np.array(batch_array)
    #print("The len of batch array is", len(np.unique(batch_array)))
    
    for i in np.unique(batch_array):
        
        a_batch_idx = np.where(batch_array== i)[0]
        
        batch = array[a_batch_idx] #[(i*batch_size):(batch_size*i+batch_size)] 
        real_life = life_array[a_batch_idx] #[(i*batch_size):(batch_size*i+batch_size)] 
        
        mean, std, minimum = np.mean(batch), np.std(batch), np.min(batch)
        life_mean = np.mean(real_life)
        
        #print(batch.shape, real_life.shape)
        baseline_eliminated = (batch.reshape(len(batch),) ) #- 0.42* real_life.reshape(len(real_life), )
        
        mean_b, std_b, minimum_b = np.mean(baseline_eliminated ), np.std(baseline_eliminated ), np.min(baseline_eliminated )
        
        baseline_eliminated_list.append(mean_b)
        
        standard1 = mean_b - (L * std_b)
        
        P1 = len(np.where(baseline_eliminated < standard1)[0]) / len(baseline_eliminated) 

        mean_life_list.append(life_mean)
        mean_list.append(mean)
        
        RUL_list.append(mean - score)
        
        if len(mean_life_list) > 1: 
            if mean_life_list[-1] > 0.98:
                score = 0
                score_list.append(score)
            
            else:          
                if P1 >= tau:
                    
                    if len(mean_life_list) > 10:            
                        
                        if len(np.where( np.diff(mean_life_list[-10:]) > 0.3)[0]) == 0 and len(np.where( np.diff(mean_life_list[-15:]) < -0.3)[0]) == 0:                
                            if (RUL_list[-1] - RUL_list[-10]) <= - 0.84:
                                print(i)
                                score += (alpha * P1 ) + 0.01
                                score_list.append(score)
                            else:
                                score +=   (alpha * P1 ) 
                                score_list.append(score)
                        else:
                            score +=  (alpha * P1 ) 
                            score_list.append(score)
                
                else :
                    score -= (alpha * P1 ) #(alpha * O)
                    score_list.append(score)        
        
        elif len(mean_life_list) <= 10:
            score = 0
            score_list.append(score)
                
    #print("score list = 0:", score_list)
    RUL_list = np.array(RUL_list) 
    return np.minimum(RUL_list, 1), mean_life_list, mean_list, baseline_eliminated_list, score_list  