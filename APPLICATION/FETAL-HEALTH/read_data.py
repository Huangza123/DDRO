# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 15:48:09 2025

@author: Huang
"""

import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch

def read_fetal_health(file_name):
    data=pd.read_csv(file_name)
    
    data0=np.array(data)[:,0:22]     
    label0=data=np.array(data)[:,-1]
    idx=np.arange(0,data.shape[0])
    
    np.random.shuffle(idx)
    save_data=data0[idx]
    save_label=label0[idx]
    # print(save_data.shape)
    2126//5
    for i in range(5):
        # 5-cross validation data
        test_data=save_data[i*425:(i+1)*425]
        test_label=save_label[i*425:(i+1)*425]
        
        np.save('./fetal_health-5-'+str(i+1)+'data-tst.npy',test_data)
        np.save('./fetal_health-5-'+str(i+1)+'label-tst.npy',test_label)
        
        train_data,train_label=[],[]
        for j in range(5):
            if j!=i:
                train_data.append(save_data[j*425:(j+1)*425])
                train_label.append(save_label[j*425:(j+1)*425])
        train_data=np.array(train_data).reshape(-1,22)
        train_label=np.array(train_label).reshape(-1)
        np.save('./fetal_health-5-'+str(i+1)+'data-tra.npy',train_data)
        np.save('./fetal_health-5-'+str(i+1)+'label-tra.npy',train_label)
    np.save('./data.npy',save_data)
    np.save('./label.npy',save_label)
def construct_cv_trainloader():
    cv_trainloader=[]
    cv_testloader=[]
    label=np.load('./label.npy', allow_pickle=True)
    # print(label)
    for i in range(5):
        train_data=np.load('./fetal_health-5-'+str(i+1)+'data-tra.npy', allow_pickle=True)
        train_label=np.load('./fetal_health-5-'+str(i+1)+'label-tra.npy', allow_pickle=True)
        test_data=np.load('./fetal_health-5-'+str(i+1)+'data-tst.npy', allow_pickle=True)
        test_label=np.load('./fetal_health-5-'+str(i+1)+'label-tst.npy', allow_pickle=True)
        
        # print(type(train_data))
        tensor_train_data=torch.from_numpy(train_data.astype(float)).float()
        mean=torch.mean(tensor_train_data)
        std=torch.std(tensor_train_data)
        tensor_train_data=(tensor_train_data-mean)/std
        tensor_train_label=torch.from_numpy(train_label.reshape(-1,1).astype(float))-1
        torch_train_dataset=Data.TensorDataset(tensor_train_data,tensor_train_label)
        trainloader=Data.DataLoader(dataset=torch_train_dataset,
                                    batch_size=128,
                                    shuffle=True)
        #testloader
        # print(test_data)
        tensor_test_data=torch.from_numpy(test_data.astype(float))
        tensor_test_data=(tensor_test_data-mean)/std
        tensor_test_label=torch.from_numpy(test_label.reshape(-1,1).astype(float))-1
        torch_test_dataset=Data.TensorDataset(tensor_test_data,tensor_test_label)
        testloader=Data.DataLoader(dataset=torch_test_dataset,
                                    batch_size=len(test_data),
                                    shuffle=True)
        cv_trainloader.append(trainloader)
        cv_testloader.append(testloader)
    attr_count=len(test_data[0])
    #print(attr_count)
    # print(count_all)
    num_classes=len(np.unique(label))
    print(np.unique(label))
    return cv_trainloader,cv_testloader,attr_count,num_classes  
    
    
    
            
            
if __name__=='__main__':
    file_name='./fetal_health.csv'
    read_fetal_health(file_name)
    cv_trainloader,cv_testloader,attr_count,num_classes=construct_cv_trainloader()
    print(attr_count,num_classes)