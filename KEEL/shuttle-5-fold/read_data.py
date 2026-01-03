# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:31:19 2021
@author: Zhan ao Huang
@email: huangzhan_ao@outlook.com
"""
import numpy as np
import os
import torch.utils.data as Data
import torch

def read_data(path):
    train_data=[[],[],[],[],[]]
    test_data=[[],[],[],[],[]]
    #attr_params=['M','F','I','negative','positive']
    #attr_params_dict={'M':'0.1','F':'0.3','I':'0.5','negative':'0','positive':'1'}
    #MIT, NUC, CYT, ME1, ME2, ME3, EXC, VAC, POX, ERL
    for name in os.listdir(path):
        if 'shuttle' in name:
            #train file
            fold=int(name[-8])
            
            if 'tra' in name:    
                f=open(path+name)
                line=f.readline()
                while line:
                    if '@data' in line:
                        line=f.readline()
                        while line:
                            # line=line.replace('MIT','0')
                            # line=line.replace('NUC','1')
                            # line=line.replace('CYT','2')
                            # line=line.replace('ME1','3')
                            # line=line.replace('ME2','4')
                            
                            # line=line.replace('ME3','5')
                            # line=line.replace('EXC','6')
                            # line=line.replace('VAC','7')
                            
                            # line=line.replace('POX','8')
                            # line=line.replace('ERL','9')
                            
                            train_data[fold-1].append(line.strip('\n').split(','))
                            line=f.readline()
                    line=f.readline()
                f.close()
            elif 'tst' in name:
                f=open(path+name)
                line=f.readline()
                while line:
                    if '@data' in line:
                        line=f.readline()
                        while line:
                            # line=line.replace('MIT','0')
                            # line=line.replace('NUC','1')
                            # line=line.replace('CYT','2')
                            # line=line.replace('ME1','3')
                            # line=line.replace('ME2','4')
                            
                            # line=line.replace('ME3','5')
                            # line=line.replace('EXC','6')
                            # line=line.replace('VAC','7')
                            
                            # line=line.replace('POX','8')
                            # line=line.replace('ERL','9')
                            
                            test_data[fold-1].append(line.strip('\n').split(','))
                            line=f.readline()
                    line=f.readline()
                f.close()
    # print(train_data)
    for i in range(5):
        for j in range(len(train_data[i])):
            for k in range(len(train_data[i][j])):
                #print(train_data[i][j][k],i,j,k)
                train_data[i][j][k]=float(train_data[i][j][k])
    for i in range(5):
        for j in range(len(test_data[i])):
            for k in range(len(test_data[i][j])):
                test_data[i][j][k]=float(test_data[i][j][k])
    return train_data,test_data
def construct_cv_trainloader(train_data_all,test_data_all):
    cv_trainloader=[]
    cv_testloader=[]
    count_all=[]
    for i in range(5):
        train_data,train_label=[],[]
        test_data,test_label=[],[]
        data_count=np.zeros(5)
        for j in range(len(train_data_all[i])):
            train_label.append(train_data_all[i][j][-1])
            train_data.append(train_data_all[i][j][0:-1])
            data_count[[1,2,3,4,5].index(train_data_all[i][j][-1])]+=1
            # if train_data_all[i][j][-1]==6 or train_data_all[i][j][-1]==7:
            #     print(train_data_all[i][j][-1])
        for k in range(len(test_data_all[i])):
            test_data.append(test_data_all[i][k][0:-1])
            test_label.append(test_data_all[i][k][-1])
        count_all.append(data_count)
        print(data_count,'datacount')
        #trainloader
        tensor_train_data=torch.from_numpy(np.array(train_data).reshape(-1,len(train_data[0]))).float()
        mean=torch.mean(tensor_train_data)
        std=torch.std(tensor_train_data)
        tensor_train_data=(tensor_train_data-mean)/std
        tensor_train_label=torch.from_numpy(np.array(train_label).reshape(-1,1))-1
        torch_train_dataset=Data.TensorDataset(tensor_train_data,tensor_train_label)
        trainloader=Data.DataLoader(dataset=torch_train_dataset,
                                    batch_size=128,
                                    shuffle=True)
        #testloader
        tensor_test_data=torch.from_numpy(np.array(test_data).reshape(-1,len(test_data[0]))).float()
        tensor_test_data=(tensor_test_data-mean)/std
        tensor_test_label=torch.from_numpy(np.array(test_label).reshape(-1,1))-1
        torch_test_dataset=Data.TensorDataset(tensor_test_data,tensor_test_label)
        testloader=Data.DataLoader(dataset=torch_test_dataset,
                                    batch_size=len(test_data),
                                    shuffle=False)
        cv_trainloader.append(trainloader)
        cv_testloader.append(testloader)
    attr_count=len(test_data[0])
    # print(attr_count)
    # print(count_all)
    num_classes=5
    return cv_trainloader,cv_testloader,attr_count,num_classes
if __name__=='__main__':
    train_data_all,test_data_all=read_data('./')
    # print(train_data_all)
    cv_trainloader,cv_testloader,attr_count,num_classes=construct_cv_trainloader(train_data_all,test_data_all)
    # print(cv_trainloader,cv_testloader,attr_count)