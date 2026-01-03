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
import matplotlib.pyplot as plt
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
        best_metrics=train_oremm(net,cv_trainloader[i],cv_testloader[i],attr_count,i+1,num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['OREMM',np.array(best_metrics_set).mean(axis=0)]))
    f.write('\n')
    f.close()
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
        print(np.unique(labels))
        # print(labels.detach().numpy().shape,outputs.detach().numpy().shape)
        if len(np.unique(labels))==4:
            #(1,3,4,5)
            #(0,2,3,4)
            print(outputs.shape,outputs[0:,1].shape,outputs[2:,4].shape)
            outputs=torch.concatenate((outputs[:,0].reshape(outputs.shape[0],-1),outputs[:,2:5].reshape(outputs.shape[0],-1)),dim=1)
            outputs_roc=torch.softmax(outputs,axis=1)
            num_classes=len(np.unique(labels))
        macro_avg_roc_auc_score=roc_auc_score(labels.detach().cpu().numpy().reshape(-1),outputs_roc.detach().cpu().numpy().reshape(-1,num_classes),average='macro',multi_class='ovo')
        # print(macro_avg_roc_auc_score,'roc-auc')
        
        macro_avg_precision_set.append(macro_avg_precision)
        macro_avg_recall_set.append(macro_avg_recall)
        macro_avg_f1_set.append(macro_avg_f1)
        _matthews_corrcoef_set.append(_matthews_corrcoef)
        macro_avg_roc_auc_score_set.append(macro_avg_roc_auc_score)
        
   
    return np.array(macro_avg_precision_set).mean(),np.array(macro_avg_recall_set).mean(),np.array(macro_avg_f1_set).mean(),np.array(_matthews_corrcoef_set).mean(),np.array(macro_avg_roc_auc_score_set).mean()

def OREMM(pos_data,neg_data,gnum=10):
    new_pos_data=[]
    sampling_count_per_data=1
    # print(sampling_count_per_data)
    gflag=True
    while gflag:
        # print(gnum,'gnum')
        for i in range(pos_data.shape[0]):
            if gflag:
                if i==0:
                    Cx=DiscoverCMR(pos_data[i], pos_data[i+1:], neg_data)
                    Ax=IdeClearReg(pos_data[i], Cx, pos_data)
                    if len(Ax)>0:
                        S_syn=Generate(pos_data[i],Ax,pos_data,neg_data,sampling_count_per_data)
                        # print(S_syn)
                        for x_syn in S_syn:
                            new_pos_data.append(x_syn)
                            gnum-=1
                            print(gnum,'gnum1')
                            if gnum==0:
                                gflag=False
                                break
                    
                elif i==pos_data.shape[0]-1:
                    Cx=DiscoverCMR(pos_data[i], pos_data[0:pos_data.shape[0]-1], neg_data)
                    # print(Cx.shape)
                    Ax=IdeClearReg(pos_data[i], Cx, pos_data)
                    if len(Ax)>0:
                        S_syn=Generate(pos_data[i],Ax,pos_data,neg_data,sampling_count_per_data)
                        for x_syn in S_syn:
                            new_pos_data.append(x_syn)
                            gnum-=1
                            print(gnum,'gnum2')
                            if gnum==0:
                                gflag=False
                                break
                else:
                    Cx=DiscoverCMR(pos_data[i], np.concatenate((pos_data[0:i],pos_data[i+1:]),axis=0), neg_data)
                    # print(Cx.shape)
                    Ax=IdeClearReg(pos_data[i], Cx, pos_data)
                    # print(len(Ax))
                    if len(Ax)>0:
                        S_syn=Generate(pos_data[i],Ax,pos_data,neg_data,sampling_count_per_data)
                        # print(S_syn.shape)
                        for x_syn in S_syn:
                            new_pos_data.append(x_syn)
                            gnum-=1
                            print(gnum,'gnum3')
                            if gnum==0:
                                gflag=False
                                break
                        # plt.scatter(neg_data[:,0],neg_data[:,1],color='white',edgecolor='green',marker='o',s=80)
                        # plt.scatter(pos_data[:,0],pos_data[:,1],color='white',edgecolor='blue',marker='o',s=80)
                        # plt.scatter(S_syn[:,0],S_syn[:,1],color='red',marker='*',s=20)
                        # plt.show()
            else:
                break
    return np.array(new_pos_data).reshape(-1,pos_data.shape[1])
def DiscoverCMR(x_pos,pos_data,neg_data,q=5):
    """q is a counting parameter used to discover candicate minority class regions"""
    data=np.concatenate((pos_data,neg_data),axis=0)
    dis=np.mean((x_pos-data)**2,axis=1)
    # print(pos_data.shape,neg_data.shape,'dis',dis.shape)
    ascending_idx=np.argsort(dis)
    
    ascending_data=data[ascending_idx]
    # print(ascending_data.shape,dis.shape)
    count,t=0,0
    for k in range(ascending_data.shape[0]):    
        #print(k)
        # print((ascending_data[k][0:2]==neg_data[0:5,0:2]).all(1),ascending_data[k][0:2], neg_data[0:5,0:2])
        if (ascending_data[k] == neg_data).all(1).any():
            count+=1
            # print(count,'count...')
            if count == q:
                t=max([1,k-q])
                break
        else:
            count=0
    # print(t,'t')
    return ascending_data[0:t]
def IdeClearReg(x_pos,Cx,pos_data):
    Ax=[]
    for p in range(Cx.shape[0]):
        x_c=(x_pos+Cx[p])/2
        r_p=np.mean((x_pos-Cx[p])**2,axis=0)/2
        # print(r_p,'rp')
        flag_clean=1
        for l in range(0,p):
            r=np.mean((x_c-Cx[l])**2,axis=0)
            # print(r,r_p,Cx[l] not in pos_data)
            if (Cx[l] not in pos_data) and (r <= r_p):
                flag_clean=0
                break
        if flag_clean:
            Ax.append(Cx[p])
        
    return Ax
def Generate(x_pos,Ax,pos_data,neg_data,gnum):
    S_syn=[]
    while gnum>0:
        x_s=Ax[np.random.randint(0,len(Ax))]
        
        gamma=np.random.uniform(0,1)
        if x_s in neg_data:
            gamma/=2
        x_syn=x_pos+gamma*(x_s-x_pos)
        S_syn.append(x_syn)
        gnum-=1
    # print(S_syn,x_pos.shape)
    return np.array(S_syn).reshape(-1,pos_data.shape[1])
        
def train_oremm(net,trainloader,testloader,attr_count,fold,num_class,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    EPOCH=1000
    optimizer=torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    
    weights=torch.zeros(num_class)
    for _,data in enumerate(trainloader):
        inputs,labels=data
        for i in range(labels.shape[0]):
            weights[int(labels[i])]+=1
    
    # metrics,best_metrics=0,[]
    # pos_data,neg_data=np.random.uniform(0,1,(10,2)),np.random.uniform(1,2,(100,2))
    # new_pos_data=OREMM(pos_data,neg_data,10)
    data_all,label_all=[],[]
    for i in range(num_class):
        data_all.append([])
        label_all.append([])
    
    for i, data in enumerate(trainloader):
        inputs,labels=data
        for _ in range(labels.shape[0]):
            data_all[int(labels[_])].append(inputs.numpy()[_])
            label_all[int(labels[_])].append(int(labels[_]))
    
    num_count=np.zeros(num_class)
    for i in range(len(label_all)):
        num_count[i]=len(label_all[i])
    neg_count=np.max(num_count)
    
    pos_label_list=[]
    resampling_data,resampling_label=[],[]
    
    for i in range(len(label_all)):
        if len(label_all[i])!=neg_count:
            pos_label_list.append(label_all[i][0])
        else:
            for j in range(len(data_all[i])):
                resampling_data.append(data_all[i][j])
                resampling_label.append(label_all[i][j])
    # print(resampling_label)
    
   
    for lb in pos_label_list:
        pos_data,neg_data=[],[]
        for i in range(len(label_all)):
            for j in range(len(label_all[i])):
                if label_all[i][j]==lb:
                    pos_data.append(data_all[i][j])
                else:
                    neg_data.append(data_all[i][j])
        sampling_count=int(neg_count-len(pos_data))
        # sampling_count=3
        
        pos_data=np.array(pos_data).reshape(-1,attr_count)
        neg_data=np.array(neg_data).reshape(-1,attr_count)
        print(lb, ' class'+ 'sampling start...')
        new_pos_data=OREMM(pos_data,neg_data,sampling_count)
        
        over_pos_data=np.concatenate((pos_data, new_pos_data),axis=0)
        over_pos_label=np.ones(len(over_pos_data))*lb
        
        # print(over_pos_data.shape,over_pos_label.shape)
        for k in range(over_pos_data.shape[0]):
            resampling_data.append(over_pos_data[k])
            resampling_label.append(over_pos_label[k])
    
    data=torch.from_numpy(np.array(resampling_data)).float()
    label=torch.from_numpy(np.array(resampling_label)).long()
    
    train_dataset=Data.TensorDataset(data,label)
    train_loader=Data.DataLoader(dataset=train_dataset,
                                batch_size=128,
                                shuffle=True)
    
    loss_func=nn.CrossEntropyLoss(weight=1/(weights+1e-8))
    metrics,best_metrics=0,[]
    
    for epoch in range(EPOCH):
        for _,data in enumerate(train_loader):
            inputs,labels=data
            outputs=net(inputs)
            loss=loss_func(outputs,labels.reshape(-1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%20==0:
            # print(epoch,loss.detach())
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,num_class,device)
            
            f=open(path+'/log_OREMM/log_OREMM.txt','a')
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
    f=open(path+'/log_OREMM/log_OREMM.txt','a')
    f.write(str(['best_metrics:',best_metrics]))      
    f.write('\n')
    return best_metrics
def train(path):
    if os.path.exists(path+'/log_OREMM/log_OREMM.txt'):
        print('Remove log_OREMM.txt')
        os.remove(path+'/log_OREMM/log_OREMM.txt') 
    elif os.path.exists(path+'/log_OREMM'):
        pass
    else:
        os.mkdir(path+'log_OREMM')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,attr_count,num_classes=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    net=Net(attr_count,num_classes)
    
    cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path)

if __name__=='__main__':
    path='../shuttle-5-fold-000/'
    train(path)
    