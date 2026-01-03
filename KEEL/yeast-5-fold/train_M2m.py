# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:42:30 2021
@author: Zhan ao Huang
@email: huangzhan_ao@outlook.com
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
import torch.utils.data as Data

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

def test(net,testloader,num_classes,device):
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
        macro_avg_roc_auc_score=roc_auc_score(labels.detach().cpu().numpy().reshape(-1),outputs_roc.detach().cpu().numpy().reshape(-1,num_classes),average='macro',multi_class='ovo')
        # print(macro_avg_roc_auc_score,'roc-auc')
        
        macro_avg_precision_set.append(macro_avg_precision)
        macro_avg_recall_set.append(macro_avg_recall)
        macro_avg_f1_set.append(macro_avg_f1)
        _matthews_corrcoef_set.append(_matthews_corrcoef)
        macro_avg_roc_auc_score_set.append(macro_avg_roc_auc_score)
        
   
    return np.array(macro_avg_precision_set).mean(),np.array(macro_avg_recall_set).mean(),np.array(macro_avg_f1_set).mean(),np.array(_matthews_corrcoef_set).mean(),np.array(macro_avg_roc_auc_score_set).mean()

def cv_train_au(netg,netf,cv_trainloader,cv_testloader,attr_count,num_classes,path):
    init_netg=copy.deepcopy(netg)
    init_netf=copy.deepcopy(netf)
    best_metrics_set=[]
    for i in range(len(cv_trainloader)):
        print('%d fold cross validation:'%(i+1))
        best_metrics=train_m2m(netg,netf,cv_trainloader[i],cv_testloader[i],attr_count,i+1,num_classes,path)
        best_metrics_set.append(best_metrics)
        netg=init_netg
        netf=init_netf
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['M2m',np.array(best_metrics_set).mean(axis=0)]))
    f.write('\n')
    f.close()
def train_m2m(netg,netf,trainloader,testloader,attr_count,fold,num_classes,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    netg.to(device)
    netf.to(device)
    
    EPOCH=1000
    weight=torch.zeros(num_classes)
    data_all=[]
    label_all=[]
    for i in range(num_classes):
        data_all.append([])
        label_all.append([])
    for _,data in enumerate(trainloader):
        inputs,labels=data
        # print(inputs.shape)
        for i in range(labels.shape[0]):
            
            weight[int(labels[i])]+=1
            # data_all[int(labels[i])].append(inputs[i].numpy())
            # label_all[int(labels[i])].append(labels[i].numpy())
            
    max_weight=torch.max(weight)
    weight_factor=(1./(weight+1e-8)).to(device)
    loss_func=nn.CrossEntropyLoss(weight=weight_factor)
    # for i in range(len(weight)):
    #     if len(data_all[i])!=max_weight:
    #         # print(len(data_all[i]),int(max_weight-weight[i]))
    #         rand_idx=np.random.randint(0,len(data_all[i]),int(max_weight-weight[i]))
            
    #         for k in rand_idx:
    #             data_all[i].append(data_all[i][k])
    #             label_all[i].append(label_all[i][k])
    
    # data_all=torch.from_numpy(np.array(data_all).reshape(-1,data_all[0][0].shape[0]))
    # label_all=torch.from_numpy(np.array(label_all).reshape(-1))

    # train_dataset=Data.TensorDataset(data_all,label_all)
    # trainloader=Data.DataLoader(dataset=train_dataset,
    #                             batch_size=128,
    #                             shuffle=True)

    
    metrics,best_metrics=0,[]
    lam=0.2
    step_size=0.01
    netg_epoch=100
    optimizerf=torch.optim.Adam(netf.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    optimizerg=torch.optim.Adam(netg.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    for epoch in range(EPOCH):
        #train netf
        if epoch <netg_epoch:
            netg.train()
            for i,(inputs,targets) in enumerate(trainloader):
                outputs=netg(inputs)
                #targets=F.one_hot(targets.long(),num_classes).reshape(-1,num_classes).float()
                lossg=loss_func(outputs,targets.long().reshape(-1))
                optimizerg.zero_grad()
                lossg.backward()
                optimizerg.step()
        else:
           #train netf
           #translate majority to minority 
           netg.eval()
           netf.eval()
           for i,(inputs,targets) in enumerate(trainloader):
               T=10
               inputs.requires_grad_(True)
               #targets=F.one_hot(targets.long(),num_classes).reshape(-1,num_classes).float()
               
               while T:
                   output_g=netg(inputs)
                   output_f=netf(inputs)
                   
                   
                   loss_g=loss_func(output_g,targets.long().reshape(-1))
                   loss_f=(output_f*targets).mean()
                   
                   loss=loss_g+loss_f*lam
                   grad,=torch.autograd.grad(loss,[inputs])
                   inputs=inputs-step_size*grad
                   inputs = torch.clamp(inputs, 0, 1)
                   T-=1
                   
               netf.train()
               inputs.detach()
               outputs=netf(inputs)
               loss_M2m=loss_func(outputs,targets.reshape(-1).long())
               optimizerf.zero_grad()
               loss_M2m.backward()
               optimizerf.step()
               
        if epoch <netg_epoch and epoch %20==0:
            
            precision,recall,f1,mcc,auc=test(netg,testloader,num_classes,device)
            f=open(path+'/log_M2m/log_M2m.txt','a')
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
            
        elif epoch >=netg_epoch and epoch %20==0:
            precision,recall,f1,mcc,auc=test(netf,testloader,num_classes,device)
            f=open(path+'/log_M2m/log_M2m.txt','a')
            
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
    f=open(path+'/log_M2m/log_M2m.txt','a')
    f.write(str(['best_metrics:',best_metrics]))      
    f.write('\n')
    return best_metrics
    
def train(path):
    if os.path.exists(path+'/log_M2m/log_M2m.txt'):
        print('Remove log_M2m.txt')
        os.remove(path+'/log_M2m/log_M2m.txt') 
    elif os.path.exists(path+'/log_M2m'):
        pass
    else:
        os.mkdir(path+'log_M2m')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,attr_count,num_classes=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    netg=Net(attr_count,num_classes)
    netf=Net(attr_count,num_classes)
    
    cv_train_au(netg,netf,cv_trainloader,cv_testloader,attr_count,num_classes,path)

if __name__=='__main__':
    path='../yeast-5-fold/'
    train(path)