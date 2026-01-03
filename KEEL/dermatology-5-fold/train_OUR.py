# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:41:17 2024

@author: Lenovo
"""

import read_data as rd
import os
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score,matthews_corrcoef,roc_auc_score
import numpy as np
import math
class Net(nn.Module):
    def __init__(self,attr_count,num_classes):
        super(Net,self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(attr_count,128),
            nn.ReLU(),
            )
        self.layer2=nn.Sequential(
            nn.Linear(128,num_classes),
            nn.ReLU()
            )
    def forward(self,x):
        x1=self.layer1(x.view(x.size(0),-1))
        x2=self.layer2(x1)
        return x1,x2
def cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path):
    init_net=copy.deepcopy(net)
    best_metrics_set=[]
    for i in range(len(cv_trainloader)):
        print('%d fold cross validation:'%(i+1))
        best_metrics=train_our(net,cv_trainloader[i],cv_testloader[i],attr_count,i+1,num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['OUR',np.array(best_metrics_set).mean(axis=0)]))
    f.write('\n')
    f.close()
def test(net,testloader,num_classes,device):
    macro_avg_precision_set,macro_avg_recall_set,macro_avg_f1_set,_matthews_corrcoef_set,macro_avg_roc_auc_score_set=[],[],[],[],[]
    for _,data in enumerate(testloader):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)
        _,outputs=net(inputs)
       
        outputs_roc=torch.softmax(outputs,axis=1)
        # print(outputs_roc)
        predicts=torch.argmax(outputs,axis=1)
        labels,predicts=labels.cpu(),predicts.cpu()
        
        macro_avg_f1=f1_score(labels,predicts,average='macro',zero_division=True)
        macro_avg_precision=precision_score(labels,predicts,average='macro',zero_division=True)
        macro_avg_recall=recall_score(labels,predicts,average='macro',zero_division=True)
        _matthews_corrcoef=matthews_corrcoef(labels,predicts)
        # print(labels.detach().numpy().shape,outputs.detach().numpy().shape)
        # if len(np.unique(labels))==4:
        #     #(0,1,3,4)
        #     outputs=torch.concatenate((outputs[:,0:2],outputs[:,3:5]),dim=1)
        #     outputs_roc=torch.softmax(outputs,axis=1)
        #     num_classes=len(np.unique(labels))
        macro_avg_roc_auc_score=roc_auc_score(labels.detach().cpu().numpy().reshape(-1),outputs_roc.detach().cpu().numpy().reshape(-1,num_classes),average='macro',multi_class='ovo')
        # print(macro_avg_roc_auc_score,'roc-auc')
        
        macro_avg_precision_set.append(macro_avg_precision)
        macro_avg_recall_set.append(macro_avg_recall)
        macro_avg_f1_set.append(macro_avg_f1)
        _matthews_corrcoef_set.append(_matthews_corrcoef)
        macro_avg_roc_auc_score_set.append(macro_avg_roc_auc_score)
        
   
    return np.array(macro_avg_precision_set).mean(),np.array(macro_avg_recall_set).mean(),np.array(macro_avg_f1_set).mean(),np.array(_matthews_corrcoef_set).mean(),np.array(macro_avg_roc_auc_score_set).mean()
def our(net,Z_B,labels,U,mu,lambda_mean,tail_class_set):
    # print(Z_B.shape,labels.shape,U.shape,'z_b')
    dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    Z_B_new=Z_B.clone()
    for i in range(Z_B.shape[1]):
        #print(labels[i].item(),tail_class_set)
        if int(labels[i].item()) in tail_class_set:
            epsilon=dist.sample([1])
            Z_B_new[:,i]=Z_B[:,i]+mu*lambda_mean*U*epsilon
    
    return net.layer2(Z_B_new.T)
def train_our(net,trainloader,testloader,attr_count,fold,num_classes,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    EPOCH=1000
    optimizer=torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    
    weights=torch.zeros(num_classes)
    bs,N,q,k,mu,lambda_mean=0,0,128,100,0.02,0
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
    
    loss_func=nn.CrossEntropyLoss(1/(weights+1e-8))
    metrics,best_metrics=0,[]
    for epoch in range(EPOCH):
        Q=torch.zeros((math.ceil(N/bs),q,q))
        Sigma=torch.zeros((q,q))
        U=torch.zeros(128,)
        for j,data in enumerate(trainloader):
            inputs,labels=data
            
            Z_B,outputs=net(inputs) #Z_B.shape,bs,width
            Z_B=Z_B.reshape(Z_B.shape[1],-1) #width, bs
            if epoch ==k-1:
                Sigma=torch.mm(Z_B, Z_B.T)
                Q[j]=Sigma
            elif epoch >=k:
                Sigma=torch.mm(Z_B, Z_B.T)
                Q[j]=Sigma
                
                # OUR operation for tail class
                outputs=our(net,Z_B,labels,U,mu,lambda_mean,tail_class_set)
            
            labels_one_hot=F.one_hot(labels.long(),num_classes).reshape(-1,num_classes).float()
            
            loss=loss_func(outputs,labels_one_hot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        Sigma_Z=torch.sum(Q,axis=0)/N
        vals,vecs=np.linalg.eig(Sigma_Z.detach().numpy())
        U=vecs[:,-1].real
        lambda_mean=vals[0:10].real.mean()
       
        if epoch%20==0:
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,num_classes,device)
            
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
    return best_metrics
def train(path):
    if os.path.exists(path+'/log_OUR/log_OUR.txt'):
        print('Remove log_OUR.txt')
        os.remove(path+'/log_OUR/log_OUR.txt') 
    elif os.path.exists(path+'/log_OUR'):
        pass
    else:
        os.mkdir(path+'log_OUR')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,attr_count,num_classes=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    net=Net(attr_count,num_classes)
    
    cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path)

if __name__=='__main__':
    path='../dermatology-5-fold/'
    train(path)
    