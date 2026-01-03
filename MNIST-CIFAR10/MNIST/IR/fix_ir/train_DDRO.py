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
        best_metrics=train_ddro(net,trainloader,testloader,attr_count,num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['DDRO',np.array(best_metrics_set).mean(axis=0)]))
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
def train_ddro(net,trainloader,testloader,num_classes,path):
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
    num_max=torch.max(weights)
    glabel_idx=[]
   
    for i in range(len(weights)):
        if weights[i]!=num_max:
            glabel_idx.append(i)
    print('num:',weights)
    metrics,best_metrics=0,[]
    weights_ratio=(1./(weights+1e-8)).to(device)
    loss_func=nn.CrossEntropyLoss(weight=weights_ratio)
    reloss_func=nn.CrossEntropyLoss()
    metrics,best_metrics=0,[]
    
    for epoch in range(EPOCH):
        if epoch<100:
            for _,data in enumerate(trainloader):
                inputs,labels=data
                inputs,labels=inputs.to(device),labels.to(device).long()
                outputs=net(inputs)
                #print(outputs.shape,labels.shape)
                loss=loss_func(outputs,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #break
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
                    
                    if int(labels[i]) in glabel_idx:
                        minority_data.append(inputs[i].numpy())
                        minority_label.append(labels[i].numpy())
            minority_data=torch.from_numpy(np.array(minority_data))
            minority_label=torch.from_numpy(np.array(minority_label))
            
            if epoch==100:
                train_data_gen,train_label_gen=[],[]
                
                seed_neighbor=torch.zeros((3,minority_data.size(2),minority_data.size(3))) #3*28*28
                #print(seed_neighbor.shape,minority_data.shape)
                
                gloss_func=nn.CrossEntropyLoss(weight=weights_ratio)
                for label_idx in glabel_idx:
                    _data=minority_data[(minority_label==torch.tensor([label_idx])).view(-1)] #674*1*28*28
                    #print(_data.shape)
                    
                    for g in tqdm(range(int(num_max-weights[label_idx])),ncols=80):
                        alpha=torch.rand((seed_neighbor.size(0),28,28),requires_grad=True) #3*1
                        goptimizer=torch.optim.Adam([alpha], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
                        seed=_data[torch.randint(0,int(weights[label_idx]),(1,))]#1*1*28*28
                        #print(seed.shape,'seed')
                        dis=torch.sum((seed.reshape(-1,28*28)-_data.reshape(-1,28*28))**2,axis=1) #674
                        
                        seed_neighbor[0]=seed #
                        dis[torch.argmin(dis)]=torch.max(dis)
                        
                        for i in range(1,seed_neighbor.size(0)):
                            min_idx=torch.argmin(dis)
                            seed_neighbor[i]=_data[torch.argmin(dis)]
                            #epsilon=dis[torch.argmin(dis)]
                            dis[torch.argmin(dis)]=torch.max(dis)
                        epsilon=torch.max(dis)
                        mean_sample=torch.mean(seed_neighbor,axis=0)#28*28
                        #print(mean_sample.shape,'mean_sample')
                        proposal_generate_sample,proposal_generate_loss=mean_sample,0
                        for gepoch in range(10):
                            #print(alpha.shape,seed_neighbor.shape,(alpha*seed_neighbor).sum(axis=0).shape)
                            generated_sample=torch.clamp((alpha*seed_neighbor).sum(axis=0),0,255).to(device) #28*28
                            
                            
                            """
                            plt.subplot(1,4,1)
                            plt.imshow(seed_neighbor[0].reshape(28,28).detach().cpu().numpy())
                            plt.subplot(1,4,2)
                            plt.imshow(seed_neighbor[1].reshape(28,28).detach().cpu().numpy())
                            plt.subplot(1,4,3)
                            plt.imshow(seed_neighbor[2].reshape(28,28).detach().cpu().numpy())
                            plt.subplot(1,4,4)
                            plt.imshow(generated_sample.reshape(28,28).detach().cpu().numpy())
                            plt.savefig('./generate/test'+'-'+str(gepoch)+'.png')
                            plt.title(str(gepoch))
                            plt.pause(0.01)
                            plt.clf()"""
                            goutput=net(generated_sample.view(-1,1,generated_sample.size(0),generated_sample.size(1)))
                            
                            glabel=torch.tensor([label_idx]).long().to(device)
                            gloss=torch.exp(-gloss_func(goutput,glabel))
                            
                            goutput_seed=net(seed.view(-1,1,28,28).to(device))
                            
                            prediction_seed=torch.argmax(goutput_seed,axis=1)
                            prediction_g=torch.argmax(goutput,axis=1)
                            
                            #dloss=torch.exp(-torch.min(torch.sum((mean_sample-generated_sample)),epsilon))
                            # dloss=torch.exp(-torch.min(torch.sum((mean_sample-generated_sample)**2),epsilon))
                            loss_all=gloss
                            #print(gloss,dloss)
                            goptimizer.zero_grad()
                            loss_all.backward()
                            goptimizer.step()
                            #print(alpha,'alpha')
                            if prediction_seed==prediction_g and proposal_generate_loss<=loss_all:
                                proposal_generate_loss=loss_all
                                proposal_generate_sample=generated_sample
                                #print(proposal_generate_loss,gepoch)
                        
                        # print('generate:',int(num_max-weights[label_idx])-len(train_data_gen))
                        train_data_gen.append(proposal_generate_sample.detach().cpu().numpy().reshape(-1,28,28))
                        train_label_gen.append(torch.tensor([label_idx]).cpu().numpy())
                        #print(proposal_generate_sample.shape,torch.tensor([label_idx]).shape,'generate')
                        #break
                for _,data in enumerate(trainloader):
                    inputs,labels=data
                    for i in range(inputs.size(0)):
                        #print(inputs[i].shape,labels[i].shape,'inputs')
                        train_data_gen.append(inputs[i].numpy())
                        train_label_gen.append(labels[i].numpy().reshape(1))
                tensor_train_data=torch.from_numpy(np.array(train_data_gen))
                tensor_train_label=torch.from_numpy(np.array(train_label_gen))
                
                torch_train_dataset=Data.TensorDataset(tensor_train_data,tensor_train_label)
                trainloader=Data.DataLoader(dataset=torch_train_dataset,
                                            batch_size=128,
                                            shuffle=True)
            else:
                for _,data in enumerate(trainloader):
                    inputs,labels=data
                    inputs,labels=inputs.to(device),labels.to(device).reshape(-1)
                    
                    #labels_one_hot=F.one_hot(labels.long(),num_classes).reshape(-1,num_classes).float()
                    outputs=net(inputs)
                    #print(outputs.shape)
                    loss=reloss_func(outputs,labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        if epoch%20==0:
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,device)
            
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
    train_ddro(net,trainloader,testloader,10,path)
if __name__=='__main__':
    path='./'
    train(path)
    