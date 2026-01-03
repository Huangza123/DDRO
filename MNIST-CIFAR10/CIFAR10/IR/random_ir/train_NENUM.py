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
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score,matthews_corrcoef,roc_auc_score
import numpy as np
import torch.utils.data as Data
from torchvision.models.resnet import ResNet, BasicBlock
from  tqdm import *
import matplotlib.pyplot as plt

def cv_train_au(net,trainloader,testloader,attr_count,num_classes,path):
    init_net=copy.deepcopy(net)
    best_metrics_set=[]
    for i in range(1):
        print('%d fold cross validation:'%(i+1))
        best_metrics=train_nenum(net,trainloader,testloader,attr_count,num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['NENUM',np.array(best_metrics_set).mean(axis=0)]))
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
    
def get_feature(module,input,output):
    print(output.shape)
    return output
        
def train_nenum(net,trainloader,testloader,num_classes,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    net.to(device)
    EPOCH=1000
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    weights=torch.zeros(num_classes)
    
    
    for _,data in enumerate(trainloader):
        inputs,labels=data
        
        for i in range(labels.shape[0]):
            weights[int(labels[i])]+=1
            
    
    #print('num:',weights)
    metrics,best_metrics=0,[]
    beta,r,tau=0.9999,1.2,0.5
    
    weights_ratio=(1./(weights+1e-8)).to(device)
    loss_func=nn.CrossEntropyLoss(weight=weights_ratio)

    
    for epoch in range(EPOCH):
        for _,data in enumerate(trainloader):
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=net(inputs)
            
            #feature=net.fc.register_forward_hook(get_feature)
            feature=inputs
            for name, module  in net.named_children():
                feature=module(feature)
                #print(feature.shape,name)
                if name=='avgpool':
                    feature=feature.reshape(-1,512)
                    break
            #print(feature.shape)
            probability=torch.softmax(outputs,dim=1)
            
            loss=0.
            count=0
            for i in np.unique(labels.cpu().numpy()):
                i=int(i)
                label_idx=(labels==i).reshape(-1)
                y_feature=feature[label_idx]
                n_y=(labels==i).sum()
                #print(y_feature.shape,'y_feature')
                mean_y=torch.mean(y_feature,axis=0)
                
                sigma_y=torch.zeros((512,512)).cuda()
                for c,mean in [(y_feature,mean_y)]:
                    s_c=torch.zeros((512,512)).cuda()
                    for sample in c:
                        
                        diff=(sample-mean_y).reshape(512,1)
                        
                        s_c+=torch.mm(diff,diff.T)
                    sigma_y+=s_c
                #print(sigma_y)    
                #sigma_y=torch.var(y_feature)
                
                
                
                ds_y=0.
                
                #gx=torch.ones(n_y)
                for z in np.unique(labels.cpu().numpy()):
                    z=int(z)
                    w_y=net.fc.weight[i]
                    # print(z,np.unique(labels.numpy()),probability)
                    if z!=i:
                        w_z=net.fc.weight[z]
                        #print(w_y.shape,w_z.shape,'wyz')
                        #unit_w_y_w_z=(w_y-w_z)/torch.sqrt(torch.sum((w_y-w_z)**2))
                        ds_yz=torch.mm(torch.mm((w_y-w_z).reshape(1,512),sigma_y+1e-8),(w_y-w_z).reshape(512,1))/(torch.norm(w_y-w_z+1e-8))**2
                        ds_y+=ds_yz
                        #print(ds_y)
                ds_y=ds_y/(len(np.unique(labels.cpu().numpy()))-1+1e-8)
                #print(ds_y)
                # print(len(np.unique(labels.numpy()))-1)
                
                for p in probability[label_idx][:,i]:
                    if p>=tau:
                       g0=(1-p)**r
                       weight_i0=1./(ds_y+1e-8)*(1.-beta)/(1.-beta**(n_y))*g0
                       loss_i0=-torch.log(p+1e-8)*weight_i0
                       #print(weight_i0,'>=tau')
                       loss+=loss_i0
                       count+=1
                    else:
                       g1=(1-probability[label_idx][:,i].mean())**r
                       weight_i1=1./(ds_y+1e-8)*(1.-beta)/(1.-beta**(n_y)+1e-8)*g1
                       loss_i1=-torch.log(p+1e-8)*weight_i1
                       #print(weight_i1,'<tau')
                       loss+=loss_i1
                       count+=1
                
            #loss=loss/count
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch%20==0:
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,device)
            
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
    torch.manual_seed(0)
    #BasicBlock=models.BasicBlock(inplanes, planes)
    layers=[1, 1, 1, 1]
    net = ResNet(BasicBlock,layers) 
    
    net.conv1=nn.Conv2d(3, 64,kernel_size=7, stride=1, padding=3, bias=False)
    
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
    test_data=np.load('./test_data_cifar10.npy')
    test_label=np.load('./test_label_cifar10.npy')
    # print(train_data.shape,train_label.shape,'xxx')
    
    
    # print(train_data.shape)
    tensor_train_data=torch.from_numpy(train_data).float().reshape(-1,3,32,32)
    tensor_train_label=torch.from_numpy(train_label)
    torch_train_dataset=Data.TensorDataset(tensor_train_data,tensor_train_label)
    trainloader=Data.DataLoader(dataset=torch_train_dataset,
                                batch_size=128,
                                shuffle=True)
    
    tensor_test_data=torch.from_numpy(test_data).float().reshape(-1,3,32,32)
    tensor_test_label=torch.from_numpy(test_label)
    torch_test_dataset=Data.TensorDataset(tensor_test_data,tensor_test_label)
    testloader=Data.DataLoader(dataset=torch_test_dataset,
                                batch_size=200,
                                shuffle=True)
    train_nenum(net,trainloader,testloader,10,path)
if __name__=='__main__':
    path='./'
    train(path)
    
