import json
import numpy as np

def data_open(directory): # '/home/alan0410/NAS_folder/Data/data/github/

    with open( directory + 'OP20-2A_end_mill_train.json', 'r') as f:
        train_json = json.load(f)

    with open(directory + 'OP20-2A_end_mill_test.json', 'r') as f:
        test_json = json.load(f)
        
        
    anomaly_train_index_final = np.array(train_json['anomaly_index_final'])
    unknown_train_index = np.array(train_json['unknown_train_index'])
    spindleload_train_data = np.array(train_json['spindleload_train_data'])
    servoload_x_train_data = np.array(train_json['servoload_x_train_data'])
    servoload_y_train_data = np.array(train_json['servoload_y_train_data'])
    servoload_z_train_data = np.array(train_json['servoload_z_train_data'])
    life_train_data = np.array(train_json[ 'life_train_data'])
    train_label_evaluation = np.array(train_json['train_label_evaluation'])
    anomaly_train_data = np.array(train_json['anomaly_train_data'])
    batch_num_train_data = np.array(train_json['batch_number'])
 
    #anomaly_index_final_test = np.array(train_json['anomaly_index_final'])
    #unknown_test_index = np.array(train_json['unknown_train_index'])
    #spindleload_test_data = np.array(train_json['spindleload_train_data'])
    #servoload_x_test_data = np.array(train_json['servoload_x_train_data'])
    #servoload_y_test_data = np.array(train_json['servoload_y_train_data'])
    #servoload_z_test_data = np.array(train_json['servoload_z_train_data'])
    #life_test_data = np.array(train_json[ 'life_train_data'])
    #test_label_evaluation = np.array(train_json['train_label_evaluation'])
    #batch_num_test_data = np.array(train_json['batch_number'])
    
    anomaly_index_final_test = np.array(test_json['anomaly_index_final_test'])
    unknown_test_index = np.array(test_json['unknown_test_index'])
    spindleload_test_data = np.array(test_json['spindleload_test_data'])
    servoload_x_test_data = np.array(test_json['servoload_x_test_data'])
    servoload_y_test_data = np.array(test_json['servoload_y_test_data'])
    servoload_z_test_data = np.array(test_json['servoload_z_test_data'])
    life_test_data = np.array(test_json[ 'life_test_data'])
    test_label_evaluation = np.array(test_json['test_label_evaluation'])
    batch_num_test_data = np.array(test_json['batch_number'])
   
    

    return anomaly_train_index_final, unknown_train_index, spindleload_train_data, servoload_x_train_data, servoload_y_train_data, servoload_z_train_data, life_train_data, train_label_evaluation, anomaly_train_data , batch_num_train_data, anomaly_index_final_test, unknown_test_index, spindleload_test_data,servoload_x_test_data,servoload_y_test_data,servoload_z_test_data,life_test_data,test_label_evaluation , batch_num_test_data 
        
    