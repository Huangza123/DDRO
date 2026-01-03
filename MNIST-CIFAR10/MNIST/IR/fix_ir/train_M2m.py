# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:42:30 2021
@author: Zhan ao Huang
@email: huangzhan_ao@outlook.com
"""

import os
# import torchvision
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
import copy

def cv_train_au(net,trainloader,testloader,attr_count,num_classes,path):
    init_net=copy.deepcopy(net)
    best_metrics_set=[]
    for i in range(1):
        print('%d fold cross validation:'%(i+1))
        best_metrics=train_ce(net,trainloader,testloader,attr_count,num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['DROLT',np.array(best_metrics_set).mean(axis=0)]))
    f.write('\n')
    f.close()
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


def train_m2m(netg,netf,trainloader,testloader,num_classes,path):
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
            #data_all[int(labels[i])].append(inputs[i].numpy())
            #label_all[int(labels[i])].append(labels[i].numpy())
            
    max_weight=torch.max(weight)
    weight_factor=1./(weight+1e-8).to(device)
    loss_func=nn.CrossEntropyLoss(weight=weight_factor)
    """for i in range(len(weight)):
        if len(data_all[i])!=max_weight:
            # print(len(data_all[i]),int(max_weight-weight[i]))
            rand_idx=np.random.randint(0,len(data_all[i]),int(max_weight-weight[i]))
            
            for k in rand_idx:
                data_all[i].append(data_all[i][k])
                label_all[i].append(label_all[i][k])
    
    data_all=torch.from_numpy(np.array(data_all).reshape(-1,1,28,28))
    label_all=torch.from_numpy(np.array(label_all).reshape(-1))

    train_dataset=Data.TensorDataset(data_all,label_all)
    trainloader=Data.DataLoader(dataset=train_dataset,
                                batch_size=128,
                                shuffle=True)

    """
    metrics,best_metrics=0,[]
    lam=0.2
    step_size=0.01
    netg_epoch=100
    optimizerf=torch.optim.Adam(netf.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    optimizerg=torch.optim.Adam(netg.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    for epoch in range(EPOCH):
        #train netf
        if epoch <netg_epoch:
            netg.train()
            for i,(inputs,targets) in enumerate(trainloader):
                inputs,targets=inputs.to(device),targets.to(device)
                outputs=netg(inputs)
                #targets=F.one_hot(targets.long(),num_classes).reshape(-1,num_classes).float()
                lossg=loss_func(outputs,targets.long())
                optimizerg.zero_grad()
                lossg.backward()
                optimizerg.step()
        else:
           #train netf
           #translate majority to minority 
           netg.eval()
           netf.eval()
           for i,(inputs,targets) in enumerate(trainloader):
               inputs,targets=inputs.to(device),targets.to(device)
               T=10
               inputs.requires_grad_(True)
               
               #targets=F.one_hot(targets.long(),num_classes).reshape(-1,num_classes).float()
               
               while T:
                   output_g=netg(inputs)
                   output_f=netf(inputs)
                   
                   
                   loss_g=loss_func(output_g,targets.long())
                   #128*10
                   loss_f=(output_f.reshape(-1,10)*targets.reshape(-1,1)).mean()
                   #print(loss_f,'loss_f')
                   loss=loss_g+loss_f*lam
                   grad,=torch.autograd.grad(loss,[inputs])
                   inputs=inputs-step_size*grad
                   inputs = torch.clamp(inputs, 0, 1)
                   T-=1
                   
               netf.train()
               inputs.detach()
               outputs=netf(inputs)
               loss_M2m=loss_func(outputs,targets.long())
               optimizerf.zero_grad()
               loss_M2m.backward()
               optimizerf.step()
               
        if epoch <netg_epoch and epoch %20==0:
            
            precision,recall,f1,mcc,auc=test(netg,testloader,device)
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
            precision,recall,f1,mcc,auc=test(netf,testloader,device)
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
    torch.manual_seed(0)
    #BasicBlock=models.BasicBlock(inplanes, planes)
    layers=[1, 1, 1, 1]
    netg = ResNet(BasicBlock,layers) 
    
    netg.conv1=nn.Conv2d(1, 64,kernel_size=7, stride=1, padding=3, bias=False)
    
    netg.inplanes=64
    netg.layer2=netg._make_layer(BasicBlock, 128, layers[1])
    netg.inplanes=128
    netg.layer3=netg._make_layer(BasicBlock, 256, layers[2])
    netg.inplanes=256
    netg.layer4=netg._make_layer(BasicBlock, 512, layers[3])
    netg.avgpool=nn.AdaptiveAvgPool2d(1)
    num_features=netg.fc.in_features
    netg.fc=nn.Linear(num_features,10)
    
    netf=copy.deepcopy(netg)
    
    train_data=np.load('./fix_ir_train_data.npy')
    train_label=np.load('./fix_ir_train_label.npy')
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
    train_m2m(netg,netf,trainloader,testloader,10,path)

if __name__=='__main__':
    path='./'
    train(path)