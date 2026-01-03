# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:41:17 2024

@author: Lenovo
"""


import os
# import torchvision
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score,matthews_corrcoef,roc_auc_score
import numpy as np
import torch.utils.data as Data
from torchvision.models.resnet import ResNet, BasicBlock
from  tqdm import *
import copy
import math

def test(net,testloader,device):
    macro_avg_precision_set,macro_avg_recall_set,macro_avg_f1_set,_matthews_corrcoef_set,macro_avg_roc_auc_score_set=[],[],[],[],[]
    for _,data in enumerate(testloader):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=net(inputs)
       
        outputs_roc=torch.softmax(outputs,axis=1)
        # print(outputs_roc)
        predicts=torch.argmax(outputs,axis=1)
        labels,predicts=labels.cpu(),predicts.cpu()
        
        macro_avg_f1=f1_score(labels,predicts,average='macro',zero_division=True)
        macro_avg_precision=precision_score(labels,predicts,average='macro',zero_division=True)
        macro_avg_recall=recall_score(labels,predicts,average='macro',zero_division=True)
        _matthews_corrcoef=matthews_corrcoef(labels,predicts)
        # print(labels.detach().numpy().shape,outputs.detach().numpy().shape)
        macro_avg_roc_auc_score=roc_auc_score(labels.detach().cpu().numpy(),outputs_roc.detach().cpu().numpy(),average='macro',multi_class='ovo')
        # print(macro_avg_roc_auc_score,'roc-auc')
        
        macro_avg_precision_set.append(macro_avg_precision)
        macro_avg_recall_set.append(macro_avg_recall)
        macro_avg_f1_set.append(macro_avg_f1)
        _matthews_corrcoef_set.append(_matthews_corrcoef)
        macro_avg_roc_auc_score_set.append(macro_avg_roc_auc_score)
        
   
    return np.array(macro_avg_precision_set).mean(),np.array(macro_avg_recall_set).mean(),np.array(macro_avg_f1_set).mean(),np.array(_matthews_corrcoef_set).mean(),np.array(macro_avg_roc_auc_score_set).mean()
def our(net,Z_B,labels,U,mu,lambda_mean,tail_class_set,device):
    # print(Z_B.shape,labels.shape,U.shape,'z_b')
    dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    Z_B_new=Z_B.clone()
    for i in range(Z_B.shape[1]):
        #print(labels[i].item(),tail_class_set)
        if int(labels[i].item()) in tail_class_set:
            epsilon=dist.sample([1])
            #print(Z_B,mu,lambda_mean.shape,U.shape,epsilon)
            Z_B_new[:,i]=Z_B[:,i]+mu*lambda_mean*(U.to(device))*(epsilon.to(device))
    
    return net.fc(Z_B_new.T)
def train_our(net,trainloader,testloader,num_classes,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    EPOCH=1000
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    
    weights=torch.zeros(num_classes)
    bs,N,q,k,mu,lambda_mean=0,0,512,100,0.02,0
    for _,data in enumerate(trainloader):
        inputs,labels=data
        if bs<inputs.shape[0]:
            bs=inputs.shape[0]
            
            # print(inputs.shape)
        N+=inputs.shape[0]
        for i in range(labels.shape[0]):
            weights[int(labels[i])]+=1
    tail_class_set=np.arange(0,num_classes)[weights<torch.max(weights)]
    # print(tail_class_set)
    metrics,best_metrics=0,[]
    weight_factor=(1./(weights+1e-8)).to(device)
    loss_func=nn.CrossEntropyLoss(weight=weight_factor)
    metrics,best_metrics=0,[]
    Sigma=torch.zeros((q,q)).to(device)
    U=torch.zeros(512,).to(device)
    for epoch in range(EPOCH):
        Q=torch.zeros((math.ceil(N/bs),q,q)).to(device)
        
        for j,data in enumerate(trainloader):
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)
            
            outputs=net(inputs) #Z_B.shape,bs,width
            
            Z_B=torch.zeros(inputs.shape[0],512).to(device)
            for name,module in net.named_children():
                Z_B=module(inputs)
                if 'avgpool' in name:
                    #print(Z_B.data.shape)
                    Z_B=Z_B.data.reshape(inputs.shape[0],net.fc.in_features)
                    break    
                inputs=Z_B
                
            
            Z_B=Z_B.reshape(512,-1) #width, bs
            if epoch ==k-1:
                Sigma=torch.mm(Z_B, Z_B.T)
                Q[j]=Sigma
            elif epoch >=k:
                Sigma=torch.mm(Z_B, Z_B.T)
                Q[j]=Sigma
                
                # OUR operation for tail class
                outputs=our(net,Z_B,labels,U,mu,lambda_mean,tail_class_set,device)
            
            #labels_one_hot=F.one_hot(labels.long(),num_classes).reshape(-1,num_classes).float()
            
            loss=loss_func(outputs,labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        Sigma_Z=torch.sum(Q,axis=0)/N
        vals,vecs=np.linalg.eig(Sigma_Z.detach().cpu().numpy())
        U=torch.from_numpy(vecs[:,-1].real).to(device)
        lambda_mean=vals[0:10].real.mean()
        #break
        if epoch%20==0:
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,device)
            
            f=open(path+'/log_OUR/log_OUR.txt','a')
            if precision*recall*f1*mcc*auc>=metrics:
                metrics=precision*recall*f1*mcc*auc
                best_metrics=[precision,recall,f1,mcc,auc]
            
            print('[%d,%d]'%(epoch,EPOCH),
                  'precision=%.4f'%precision,
                  'recall=%.4f'%recall,
                  'f1=%.4f'%f1,
                  'mcc=%.4f'%mcc,
                  'auc=%.4f'%auc
                  )
            
            f.write(str(['[%d,%d]'%(epoch,EPOCH),
                  'precision=%.4f'%precision,
                  'recall=%.4f'%recall,
                  'f1=%.4f'%f1,
                  'mcc=%.4f'%mcc,
                  'auc=%.4f'%auc]))      
            f.write('\n')
            f.close()
    f=open(path+'/log_OUR/log_OUR.txt','a')
    f.write(str(['best_metrics:',best_metrics]))      
    f.write('\n')
def train(path):
    if os.path.exists(path+'/log_OUR/log_OUR.txt'):
        print('Remove log_OUR.txt')
        os.remove(path+'/log_OUR/log_OUR.txt') 
    elif os.path.exists(path+'/log_OUR'):
        pass
    else:
        os.mkdir(path+'log_OUR')
    torch.manual_seed(0)
    #BasicBlock=models.BasicBlock(inplanes, planes)
    layers=[1, 1, 1, 1]
    net = ResNet(BasicBlock,layers) 
    
    net.conv1=nn.Conv2d(1, 64,kernel_size=7, stride=1, padding=3, bias=False)
    
    net.inplanes=64
    net.layer2=net._make_layer(BasicBlock, 128, layers[1])
    net.inplanes=128
    net.layer3=net._make_layer(BasicBlock, 256, layers[2])
    net.inplanes=256
    net.layer4=net._make_layer(BasicBlock, 512, layers[3])
    net.avgpool=nn.AdaptiveAvgPool2d(1)
    
    num_features=net.fc.in_features
    net.fc=nn.Linear(num_features,10)

    train_data=np.load('./random_ir_train_data.npy')
    train_label=np.load('./random_ir_train_label.npy')
    # print(train_data.shape,train_label.shape)
    test_data=np.load('./test_data.npy')
    test_label=np.load('./test_label.npy')
    # print(train_data.shape,train_label.shape,'xxx')
    
    # print(train_data.shape)
    tensor_train_data=torch.from_numpy(train_data).float().reshape(-1,1,28,28)
    tensor_train_label=torch.from_numpy(train_label)
    torch_train_dataset=Data.TensorDataset(tensor_train_data,tensor_train_label)
    trainloader=Data.DataLoader(dataset=torch_train_dataset,
                                batch_size=128,
                                shuffle=True)
    
    tensor_test_data=torch.from_numpy(test_data).float().reshape(-1,1,28,28)
    tensor_test_label=torch.from_numpy(test_label)
    torch_test_dataset=Data.TensorDataset(tensor_test_data,tensor_test_label)
    testloader=Data.DataLoader(dataset=torch_test_dataset,
                                batch_size=128,
                                shuffle=False)
    train_our(net,trainloader,testloader,10,path)

if __name__=='__main__':
    path='./'
    train(path)
    