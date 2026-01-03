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
import scipy.optimize as sopt
class Net(nn.Module):
    def __init__(self,attr_count,num_classes):
        super(Net,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(attr_count,128),
            nn.ReLU(),
            nn.Linear(128,num_classes),
            nn.ReLU(),
            )
    def forward(self,x):
        return self.layer(x.view(x.size(0),-1))

def cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path):
    init_net=copy.deepcopy(net)
    best_metrics_set=[]
    for i in range(len(cv_trainloader)):
        print('%d fold cross validation:'%(i+1))
        best_metrics=train_doro(net,cv_trainloader[i],cv_testloader[i],attr_count,i+1,num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['DORO',np.array(best_metrics_set).mean(axis=0)]))
    f.write('\n')
    f.close()
def test(net,testloader,num_classes,device):
    macro_avg_precision_set,macro_avg_recall_set,macro_avg_f1_set,_matthews_corrcoef_set,macro_avg_roc_auc_score_set=[],[],[],[],[]
    for _,data in enumerate(testloader):
        inputs,labels=data
        inputs,labels=inputs.to(device).float(),labels.to(device)
        outputs=net(inputs)
       
        outputs_roc=torch.softmax(outputs,axis=1)
        # print(outputs_roc)
        predicts=torch.argmax(outputs,axis=1)
        labels,predicts=labels.cpu(),predicts.cpu()
        
        macro_avg_f1=f1_score(labels,predicts,average='macro',zero_division=True)
        macro_avg_precision=precision_score(labels,predicts,average='macro',zero_division=True)
        macro_avg_recall=recall_score(labels,predicts,average='macro',zero_division=True)
        _matthews_corrcoef=matthews_corrcoef(labels,predicts)
        # print(outputs_roc,outputs,labels)
        # print(labels.shape,outputs_roc.shape,num_classes)
        # if len(np.unique(labels))==4:
        #     #(0,1,3,4)
        #     outputs=torch.concatenate((outputs[:,0:2],outputs[:,3:5]),dim=1)
        #     outputs_roc=torch.softmax(outputs,axis=1)
        #     num_classes=len(np.unique(labels))
        macro_avg_roc_auc_score=roc_auc_score(labels.detach().cpu().numpy().reshape(-1),outputs_roc.detach().cpu().numpy().reshape(-1,num_classes),average='macro',multi_class='ovo',labels=np.unique(labels))
        # print(macro_avg_roc_auc_score,'roc-auc')
        
        macro_avg_precision_set.append(macro_avg_precision)
        macro_avg_recall_set.append(macro_avg_recall)
        macro_avg_f1_set.append(macro_avg_f1)
        _matthews_corrcoef_set.append(_matthews_corrcoef)
        macro_avg_roc_auc_score_set.append(macro_avg_roc_auc_score)

    return np.array(macro_avg_precision_set).mean(),np.array(macro_avg_recall_set).mean(),np.array(macro_avg_f1_set).mean(),np.array(_matthews_corrcoef_set).mean(),np.array(macro_avg_roc_auc_score_set).mean()
def train_doro(net,trainloader,testloader,attr_count,fold,num_classes,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    EPOCH=1000
    optimizer=torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    
    weights=torch.zeros(num_classes)
    for _,data in enumerate(trainloader):
        inputs,labels=data
        for i in range(labels.shape[0]):
            weights[int(labels[i])]+=1
    # print(weights)
    # loss_func=nn.CrossEntropyLoss(weight=1/weights)
    # loss_func=nn.CrossEntropyLoss()
    metrics,best_metrics=0,[]
    
    weights_ratio=(1./(weights+1e-8)).to(device)
    loss_func=nn.CrossEntropyLoss(reduction='none',weight=weights_ratio)
    metrics,best_metrics=0,[]
    alpha,eps=0.1,0.01
    gamma = eps + alpha * (1 - eps)
    max_l = 10.
    C = math.sqrt(1 + (1 / alpha - 1) ** 2)
    for epoch in range(EPOCH):
        for _,data in enumerate(trainloader):
            inputs,labels=data
            batch_size = len(inputs)
            #labels_one_hot=F.one_hot(labels.long(),num_classes).reshape(-1,num_classes).float()
            # print(labels[0],labels_one_hot[0])
            outputs=net(inputs)
            # print(outputs.shape,labels_one_hot.shape)
            loss=loss_func(outputs,labels.reshape(-1).long())
            #CVaR DORO
            n1 = int(gamma * batch_size)
            n2 = int(eps * batch_size)
            rk = torch.argsort(loss, descending=True)
            loss = loss[rk[n2:n1]].sum() / alpha / (batch_size - n2)
            
            #Chi^2-DORO
            # n = int(eps * batch_size)
            # rk = torch.argsort(loss, descending=True)
            # l0 = loss[rk[n:]]
            # foo = lambda eta: C * math.sqrt((F.relu(l0 - eta) ** 2).mean().item()) + eta
            # opt_eta = sopt.brent(foo, brack=(0, max_l))
            # loss = C * torch.sqrt((F.relu(l0 - opt_eta) ** 2).mean()) + opt_eta
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%20==0:
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,num_classes,device)
            
            f=open(path+'/log_DORO/log_DORO.txt','a')
            if precision*recall*f1*mcc*auc>=metrics:
                metrics=precision*recall*f1*mcc*auc
                best_metrics=[precision,recall,f1,mcc,auc]
            
            print('[%d,%d]'%(epoch,EPOCH),'loss=%.3f'%loss.item(),
                  'precision=%.4f'%precision,
                  'recall=%.4f'%recall,
                  'f1=%.4f'%f1,
                  'mcc=%.4f'%mcc,
                  'auc=%.4f'%auc
                  )
            
            f.write(str(['[%d,%d]'%(epoch,EPOCH),'loss=%.3f'%loss.item(),
                  'precision=%.4f'%precision,
                  'recall=%.4f'%recall,
                  'f1=%.4f'%f1,
                  'mcc=%.4f'%mcc,
                  'auc=%.4f'%auc]))      
            f.write('\n')
            f.close()
    f=open(path+'/log_DORO/log_DORO.txt','a')
    f.write(str(['best_metrics:',best_metrics]))      
    f.write('\n')
    return best_metrics
def train(path):
    if os.path.exists(path+'/log_DORO/log_DORO.txt'):
        print('Remove log_DORO.txt')
        os.remove(path+'/log_DORO/log_DORO.txt') 
    elif os.path.exists(path+'/log_DORO'):
        pass
    else:
        os.mkdir(path+'log_DORO')
    # train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,attr_count,num_classes=rd.construct_cv_trainloader()
    
    torch.manual_seed(0)
    net=Net(attr_count,num_classes)
    
    cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path)

if __name__=='__main__':
    path='../fetal_health/'
    train(path)
    