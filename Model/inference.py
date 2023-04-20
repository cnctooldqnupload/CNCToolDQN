from tqdm import tqdm
import numpy as np
import torch

def inference(spindleload_data, servoload_x_data, servoload_y_data, servoload_z_data, life_data, model):    
    
    length = spindleload_data.shape[0]
    mok = length //100 
    nameoji = length % 100
    result_list = []
    
    for i in tqdm(range(mok+1)):
        spindleload_data_batch = spindleload_data[(i*100):(100*i+100)]
        servoload_x_data_batch = servoload_x_data[(i*100):(100*i+100)]
        servoload_y_data_batch= servoload_y_data[(i*100):(100*i+100)]
        servoload_z_data_batch= servoload_z_data[(i*100):(100*i+100)]
        life_data_batch = life_data[(i*100):(100*i+100)]
        
        with torch.no_grad():
            result = model.forward(spindleload_data_batch,servoload_x_data_batch, servoload_y_data_batch,  servoload_z_data_batch, life_data_batch )[1].squeeze(1)[:,0]
        
        if i == (mok):
            spindleload_data_batch = spindleload_data[(i*100):(100*i+nameoji+1)]
            servoload_x_data_batch = servoload_x_data[(i*100):(100*i+nameoji+1)]
            servoload_y_data_batch= servoload_y_data[(i*100):(100*i+nameoji+1)]
            servoload_z_data_batch= servoload_z_data[(i*100):(100*i+nameoji+1)]
            life_data_batch = life_data[(i*100):(100*i+nameoji+1)]
            
            with torch.no_grad():
                result = model.forward(spindleload_data_batch,servoload_x_data_batch, servoload_y_data_batch,  servoload_z_data_batch, life_data_batch)[1].squeeze(1)[:,0]
        
        result_list.append(result.tolist()) 

    result_list_flat = np.array([item for sublist in result_list for item in sublist]).reshape(-1, 1) 
    
    return result_list_flat