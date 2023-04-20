import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

def scoring_train_data(model, spindleload_train_data, servox_train_data, servoy_train_data, servoz_train_data, life_train_data, train_label):
    
    train_length = spindleload_train_data.shape[0]
    mok = train_length //100 
    nameoji = train_length %100 
    
    train_result_list = []

    for i in range(mok+1):
        spindleload_train_data_batch = spindleload_train_data[(i*100):(100*i+100)]
        servox_train_data_batch = servox_train_data[(i*100):(100*i+100)]
        servoy_train_data_batch= servoy_train_data[(i*100):(100*i+100)]
        servoz_train_data_batch= servoz_train_data[(i*100):(100*i+100)]
        life_train_data_batch = life_train_data[(i*100):(100*i+100)]
        result = model.forward(spindleload_train_data_batch,servox_train_data_batch, servoy_train_data_batch,  servoz_train_data_batch, life_train_data_batch )[1].squeeze(1)[:,1]

        if i == (mok):
            spindleload_train_data_batch = spindleload_train_data[(i*100):(100*i+nameoji+1)]
            servox_train_data_batch = servox_train_data[(i*100):(100*i+nameoji+1)]
            servoy_train_data_batch= servoy_train_data[(i*100):(100*i+nameoji+1)]
            servoz_train_data_batch= servoz_train_data[(i*100):(100*i+nameoji+1)]
            life_train_data_batch = life_train_data[(i*100):(100*i+nameoji+1)]
            result = model.forward(spindleload_train_data_batch,servox_train_data_batch,servoy_train_data_batch, servoz_train_data_batch, life_train_data_batch )[1].squeeze(1)[:,1]

        train_result_list.append(result.tolist())
    
    result_list_flat_train = np.array([item for sublist in train_result_list for item in sublist])    
    threshold_list = np.arange(0.05, 0.95, 0.025)
    f1_score_list = []
    
    
    scaler = MinMaxScaler( feature_range=(0, 1))
    result_list_flat_train = scaler.fit_transform(np.array(result_list_flat_train).reshape(-1, 1) ).reshape(len(result_list_flat_train))
    #result_list_flat_train = 1- result_list_flat_train    
    #print(result_list_flat_train)
    
    for threshold in threshold_list:
        result_list_flat_train_new = np.where(np.array(result_list_flat_train) > threshold , 1, 0)
        f1 = f1_score(train_label, result_list_flat_train_new, average= 'macro')
        #onfusion = confusion_matrix(train_label, result_list_flat_train)
        f1_score_list.append(f1)
    
    best_threshold = threshold_list[np.argmax(f1_score_list)]
    
    print("The best thershold: ", best_threshold )
    print("f1_score: ", np.max(f1_score_list))
    
    return best_threshold 