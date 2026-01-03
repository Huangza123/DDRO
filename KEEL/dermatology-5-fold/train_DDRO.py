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
import matplotlib.pyplot as plt
import torch.utils.data as Data

class Net(nn.Module):
    def __init__(self,attr_count,num_classes):
        super(Net,self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(attr_count,128),
            nn.ReLU(),
            nn.Linear(128,num_classes),
            nn.ReLU()
            )
    def forward(self,x):
        # print(x.view(x.size(0),-1).shape)
        return self.layer(x.view(x.size(0),-1))

def cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path):
    init_net=copy.deepcopy(net)
    best_metrics_set=[]
    for i in range(len(cv_trainloader)):
        print('%d fold cross validation:'%(i+1))
        best_metrics=train_ddro(net,cv_trainloader[i],cv_testloader[i],attr_count,i+1,num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['DDRO',np.array(best_metrics_set).mean(axis=0)]))
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
def train_ddro(net,trainloader,testloader,attr_count,fold,num_classes,path):
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
            
    num_max=torch.max(weights)
    glabel_idx=[]
   
    for i in range(len(weights)):
        if weights[i]!=num_max:
            glabel_idx.append(i)
    print('num:',weights,glabel_idx)
    weights_ratio=(1./(weights+1e-8)).to(device)
    loss_func=nn.CrossEntropyLoss(weight=weights_ratio)
    reloss_func=nn.CrossEntropyLoss()
    metrics,best_metrics=0,[]
    
    for epoch in range(EPOCH):
        if epoch<100:
            for _,data in enumerate(trainloader):
                inputs,labels=data
                #labels_one_hot=F.one_hot(labels.long(),num_classes).reshape(-1,num_classes).float()
                outputs=net(inputs)
                # print(inputs.shape)
                loss=loss_func(outputs,labels.reshape(-1).long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            """Data Driven Robust Optimization for minority samples
            generated label index:glabel_idx; num of generation: glabel_gen_num
            
            Generate new data to expand minority regions.
            \max loss to generate new sample of x, using L2 distance to control the generation position
            """
            minority_data,minority_label=[],[]
            for _,data in enumerate(trainloader):
                inputs,labels=data
                for i in range(inputs.size(0)):
                    # print(labels[i],glabel_idx)
                    if int(labels[i]) in glabel_idx:
                        minority_data.append(inputs[i].numpy())
                        minority_label.append(labels[i].numpy())
            minority_data=torch.from_numpy(np.array(minority_data))
            minority_label=torch.from_numpy(np.array(minority_label))
            
            if epoch==100:
                train_data_gen,train_label_gen=[],[]
                
                seed_neighbor=torch.zeros((3,minority_data.size(1)))
                alpha=torch.rand((seed_neighbor.size(0),seed_neighbor.size(1)),requires_grad=True)
                goptimizer=torch.optim.Adam([alpha], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
                gloss_func=nn.CrossEntropyLoss(weight=weights_ratio)
                for label_idx in glabel_idx:
                    _data=minority_data[(minority_label==torch.tensor([label_idx])).view(-1)]
                    for g in range(int(num_max-weights[label_idx])):
                        
                        seed=_data[np.random.randint(0,weights[label_idx])]
                 
                        dis=torch.sum((seed-_data)**2,axis=1)
                        seed_neighbor[0]=seed
                        dis[torch.argmin(dis)]=torch.max(dis)
                        
                        for i in range(1,seed_neighbor.size(0)):
                            min_idx=torch.argmin(dis)
                            seed_neighbor[i]=_data[torch.argmin(dis)]
                            #epsilon=dis[torch.argmin(dis)]
                            dis[torch.argmin(dis)]=torch.max(dis)
                        epsilon=torch.max(dis)
                        mean_sample=torch.mean(seed_neighbor,axis=0)
                        
                        proposal_generate_sample,proposal_generate_loss=mean_sample,0
                        for gepoch in range(10):
                            generated_sample=torch.clamp((alpha*seed_neighbor).sum(axis=0),0,1)
                            # generated_sample=(alpha*seed_neighbor).sum(axis=0)
                            # print(alpha)
                            # print(generated_sample.shape)
                            # plt.subplot(1,4,1)
                            # plt.imshow(seed_neighbor[0].reshape(1,generated_sample.shape[0]).detach().cpu().numpy())
                            # plt.subplot(1,4,2)
                            # plt.imshow(seed_neighbor[1].reshape(1,generated_sample.shape[0]).detach().cpu().numpy())
                            # plt.subplot(1,4,3)
                            # plt.imshow(seed_neighbor[2].reshape(1,generated_sample.shape[0]).detach().cpu().numpy())
                            # plt.subplot(1,4,4)
                            # plt.imshow(generated_sample.reshape(1,generated_sample.shape[0]).detach().cpu().numpy())
                            # plt.savefig('./generate/test'+'-'+str(gepoch)+'.png')
                            # plt.title(str(gepoch))
                            # plt.pause(0.01)
                            # plt.clf()
                            goutput=net(generated_sample.view(-1,generated_sample.size(0)))
                            
                            #glabel=F.one_hot(torch.tensor([label_idx]).long(),num_classes).reshape(-1,num_classes).float()
                            glabel=torch.tensor([label_idx]).long()
                            gloss=torch.exp(-gloss_func(goutput,glabel))
                            
                            goutput_seed=net(seed.view(-1,seed.size(0)))
                            
                            prediction_seed=torch.argmax(goutput_seed,axis=1)
                            prediction_g=torch.argmax(goutput,axis=1)
                            
                            loss_all=gloss
                            #print(gloss,dloss)
                            goptimizer.zero_grad()
                            loss_all.backward()
                            goptimizer.step()
                            
                            if prediction_seed==prediction_g and proposal_generate_loss<=loss_all:
                                proposal_generate_loss=loss_all
                                proposal_generate_sample=generated_sample
                                print(proposal_generate_loss,gepoch)
                        
                        # print('generate:',int(num_max-weights[label_idx])-len(train_data_gen))
                        train_data_gen.append(proposal_generate_sample.detach().numpy())
                        train_label_gen.append(torch.tensor([label_idx]).numpy())
                        #print('generate:',int(num_max-weights[label_idx])-len(train_data_gen))
                            
                        # else:
                        #     train_data_gen.append(generated_sample.detach().numpy())
                        #     train_label_gen.append(torch.tensor([label_idx]).numpy())
                for _,data in enumerate(trainloader):
                    inputs,labels=data
                    for i in range(inputs.size(0)):
                        train_data_gen.append(inputs[i].numpy())
                        train_label_gen.append(labels[i].numpy())
                tensor_train_data=torch.from_numpy(np.array(train_data_gen))
                tensor_train_label=torch.from_numpy(np.array(train_label_gen))
                
                torch_train_dataset=Data.TensorDataset(tensor_train_data,tensor_train_label)
                trainloader=Data.DataLoader(dataset=torch_train_dataset,
                                            batch_size=128,
                                            shuffle=True)
            else:
                for _,data in enumerate(trainloader):
                    inputs,labels=data
                    #labels_one_hot=F.one_hot(labels.long(),num_classes).reshape(-1,num_classes).float()
                    outputs=net(inputs)
                    # print(outputs)
                    loss=reloss_func(outputs,labels.reshape(-1).long())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                        
        if epoch%20==0:
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,num_classes,device)
            
            f=open(path+'/log_DDRO/log_DDRO.txt','a')
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
    f=open(path+'/log_DDRO/log_DDRO.txt','a')
    f.write(str(['best_metrics:',best_metrics]))      
    f.write('\n')
    return best_metrics
def train(path):
    if os.path.exists(path+'/log_DDRO/log_DDRO.txt'):
        print('Remove log_DDRO.txt')
        os.remove(path+'/log_DDRO/log_DDRO.txt') 
    elif os.path.exists(path+'/log_DDRO'):
        pass
    else:
        os.mkdir(path+'log_DDRO')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,attr_count,num_classes=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    net=Net(attr_count,num_classes)
    
    cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path)

if __name__=='__main__':
    path='../dermatology-5-fold/'
    train(path)
    