# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:41:17 2024

@author: Lenovo
"""


import os
import read_data as rd
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score,matthews_corrcoef,roc_auc_score
import numpy as np
import torch.utils.data as Data
from torchvision.models.resnet import ResNet, BasicBlock
from  tqdm import *

class Net(nn.Module):
    def __init__(self,attr_count,num_classes):
        super(Net,self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(attr_count,128),
            nn.ReLU(),
            )
        self.layer2=nn.Sequential(
            nn.Linear(128,num_classes),
            nn.ReLU(),
            )
    def forward(self,x):
        x0=self.layer1(x.view(x.size(0),-1))
        x1=self.layer2(x0)
        return x0,x1
    
def cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path):
    init_net=copy.deepcopy(net)
    best_metrics_set=[]
    for i in range(len(cv_trainloader)):
        print('%d fold cross validation:'%(i+1))
        best_metrics=train_nenum(net,cv_trainloader[i],cv_testloader[i],attr_count,i+1,num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['NENUM',np.array(best_metrics_set).mean(axis=0)]))
    f.write('\n')
    f.close()
def test(net,testloader,num_classes,device):
    macro_avg_precision_set,macro_avg_recall_set,macro_avg_f1_set,_matthews_corrcoef_set,macro_avg_roc_auc_score_set=[],[],[],[],[]
    for _,data in enumerate(testloader):
        inputs,labels=data
        # print(np.unique(labels))
        inputs,labels=inputs.to(device).float(),labels.to(device)
        _,outputs=net(inputs)
       
        outputs_roc=torch.softmax(outputs,axis=1)
        
        # print(outputs)
        predicts=torch.argmax(outputs,axis=1)
        labels,predicts=labels.cpu(),predicts.cpu()
        
        macro_avg_f1=f1_score(labels,predicts,average='macro',zero_division=True)
        macro_avg_precision=precision_score(labels,predicts,average='macro',zero_division=True)
        macro_avg_recall=recall_score(labels,predicts,average='macro',zero_division=True)
        _matthews_corrcoef=matthews_corrcoef(labels,predicts)
        # print(labels.detach().numpy().shape,outputs.detach().numpy().shape)
        # print(np.unique(labels))
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

def DS(net,inputs):
    pass
def train_nenum(net,trainloader,testloader,attr_count,fold,num_classes,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    EPOCH=1000
    optimizer=torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    
    """"net.layer2[0].weight.shape:3*128,
    net.layer2[0].bias.shape:3"""
    
    metrics,best_metrics=0,[]
    beta,r,tau=0.9999,1.2,0.5
    
    for epoch in range(EPOCH):
        for _,data in enumerate(trainloader):
            inputs,labels=data
            feature,outputs=net(inputs)
            probability=torch.softmax(outputs,dim=1)
            
            loss=0.
            # print(np.unique(labels.numpy()))
            for i in np.unique(labels.numpy()):
                i=int(i)
                label_idx=(labels==i).reshape(-1)
                y_feature=feature[label_idx]
                n_y=(labels==i).sum()
                mean_y=torch.mean(y_feature,axis=0)
                
                sigma_y=torch.zeros((128,128))
                for c,mean in [(y_feature,mean_y)]:
                    s_c=torch.zeros((128,128))
                    for sample in c:
                        
                        diff=(sample-mean_y).reshape(128,1)
                        
                        s_c+=torch.mm(diff,diff.T)
                    sigma_y+=s_c
                
                ds_y=0.
                
                for z in np.unique(labels.cpu().numpy()):
                    z=int(z)
                    w_y=net.layer2[0].weight[i]
                    
                    if z!=i:
                        w_z=net.layer2[0].weight[z]
                       
                        ds_yz=torch.mm(torch.mm((w_y-w_z).reshape(1,128),sigma_y+1e-8),(w_y-w_z).reshape(128,1))/((torch.norm(w_y-w_z+1e-8))**2)
                        ds_y+=ds_yz
                        # print(torch.norm(w_y-w_z),'norm')
                ds_y=ds_y/(len(np.unique(labels.cpu().numpy()))-1+1e-8)
                for p in probability[label_idx][:,i]:
                    if p>=tau:
                        g0=(1-p)**r
                       
                        weight_i0=1./(ds_y+1e-8)*(1.-beta)/(1.-beta**(n_y)+1e-8)*g0
                        loss_i0=-torch.log(p++1e-8)*weight_i0
                        loss+=loss_i0
                    
                              
                    else:
                        g1=(1-probability[label_idx][:,i].mean())**r
                       
                        weight_i1=1./(ds_y+1e-8)*(1.-beta)/(1.-beta**(n_y)+1e-8)*g1
                        loss_i1=-torch.log(p++1e-8)*weight_i1
                        loss+=loss_i1
                    
                     
                       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%20==0:
            # print(epoch,loss.detach())
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,num_classes,device)
            
            f=open(path+'/log_NENUM/log_NENUM.txt','a')
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
    f=open(path+'/log_NENUM/log_NENUM.txt','a')
    f.write(str(['best_metrics:',best_metrics]))      
    f.write('\n')
    return best_metrics
def train(path):
    if os.path.exists(path+'/log_NENUM/log_NENUM.txt'):
        print('Remove log_NENUM.txt')
        os.remove(path+'/log_NENUM/log_NENUM.txt') 
    elif os.path.exists(path+'/log_NENUM'):
        pass
    else:
        os.mkdir(path+'log_NENUM')
    # train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,attr_count,num_classes=rd.construct_cv_trainloader()
    
    torch.manual_seed(0)
    net=Net(attr_count,num_classes)
    
    cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path)

if __name__=='__main__':
    path='../fetal_health/'
    train(path)
    