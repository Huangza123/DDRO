# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:20:44 2024

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
import torch.utils.data as Data
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
class DROLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, class_weights=None, epsilons=None):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.class_weights = class_weights
        self.epsilons = epsilons

    def pairwise_euaclidean_distance(self, x, y):
        return torch.cdist(x, y)

    def pairwise_cosine_sim(self, x, y):
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)
        return torch.matmul(x, y.T)

    def forward(self, batch_feats, batch_targets, centroid_feats, centroid_targets):
        device = (torch.device('cuda')
                  if centroid_feats.is_cuda
                  else torch.device('cpu'))

        classes, positive_counts = torch.unique(batch_targets, return_counts=True)
        centroid_classes = torch.unique(centroid_targets).long()
        train_prototypes = torch.stack([centroid_feats[torch.where(centroid_targets == c)[0]].mean(0)
                                        for c in centroid_classes])
        pairwise = -1 * self.pairwise_euaclidean_distance(train_prototypes, batch_feats)

        # epsilons
        if self.epsilons is not None:
            mask = torch.eq(centroid_classes.contiguous().view(-1, 1), batch_targets.contiguous().view(-1, 1).T).to(
                device)
            a = pairwise.clone()
            batch_targets=batch_targets.long()
            
            pairwise[mask] = a[mask] - self.epsilons[batch_targets].to(device)

        logits = torch.div(pairwise, self.temperature)

        # compute log_prob
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
        log_prob = torch.stack([log_prob[:, torch.where(batch_targets == c)[0]].mean(1) for c in classes], dim=1)

        # compute mean of log-likelihood over positive
        mask = torch.eq(centroid_classes.contiguous().view(-1, 1), classes.contiguous().view(-1, 1).T).float().to(
            device)
        log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob_pos
        # weight by class weight
        if self.class_weights is not None:
            weights = self.class_weights[centroid_classes]
            weighted_loss = loss * weights
            loss = weighted_loss.sum() / weights.sum()
        else:
            loss = loss.sum() / len(classes)

        return loss

def cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path):
    init_net=copy.deepcopy(net)
    best_metrics_set=[]
    for i in range(len(cv_trainloader)):
        print('%d fold cross validation:'%(i+1))
        best_metrics=train_drolt(net,cv_trainloader[i],cv_testloader[i],attr_count,i+1,num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['DROLT',np.array(best_metrics_set).mean(axis=0)]))
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
def centroid(net,trainloader,feat_dim=128):
    features=torch.empty((0,feat_dim))
    targets=torch.empty(0,dtype=torch.long)
    with torch.no_grad():
        for i,(inputs,target) in enumerate(trainloader):
            feats,outputs=net(inputs)
            features=torch.cat((features,feats))
            targets=torch.cat((targets,target))
    return features,targets
def freeze_layers(net,fe_bool=True,cls_bool=True):
    if fe_bool:
        net.layer1.train()
    else:
        net.layer1.eval()
        
    if cls_bool:
        net.layer2.train()
    else:
        net.layer2.eval()
        
    for name,params in net.named_parameters():
        # print(name,params)
        if 'layer2' in name:
            params.requires_grad=cls_bool
        else:
            params.requires_grad=fe_bool
def train_drolt(net,trainloader,testloader,attr_count,fold,num_classes,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:1')
    net.to(device)
    
    EPOCH=1000
    
    feat_params = []
    feat_params_names = []
    cls_params = []
    cls_params_names = []
    learnable_epsilons = torch.nn.Parameter(torch.ones(num_classes))
    
    for name, params in net.named_parameters():
        # print(name,params.requires_grad,'tt')
        # if params.requires_grad:
        if 'layer2' in name:
            cls_params_names += [name]
            cls_params += [params]
            # print(cls_params,'cls',name)
        else:
            feat_params_names += [name]
            feat_params += [params]
            # print(feat_params,'feat',name)
    
    feat_optim = torch.optim.Adam(feat_params + [learnable_epsilons], lr=0.01,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    cls_optim = torch.optim.Adam(cls_params, lr=0.01,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    
    #resampling
    weight=torch.zeros(num_classes)
    # data_all=[]
    # label_all=[]
    # for i in range(num_classes):
    #     data_all.append([])
    #     label_all.append([])
    for _,data in enumerate(trainloader):
        inputs,labels=data
        # print(inputs.shape)
        for i in range(labels.shape[0]):
            
            weight[int(labels[i])]+=1
            # data_all[int(labels[i])].append(inputs[i].numpy())
            # label_all[int(labels[i])].append(labels[i].numpy())
            
    max_weight=torch.max(weight)
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
    weight_factor=(1./(weight+1e-8)).to(device)
    train_feat_losses=[nn.CrossEntropyLoss(weight=weight_factor),DROLoss(temperature=1,
                                                   class_weights=weight_factor,
                                                   epsilons=learnable_epsilons)]
    train_class_losses=[nn.CrossEntropyLoss(weight=weight_factor)]
    
    metrics,best_metrics=0,[]
    
    feature_extractor_duration=100
    for epoch in range(EPOCH):
        # if epoch%10==0:
        features,targets=centroid(net, trainloader)
        cls_loss=0.
        
        if epoch<feature_extractor_duration:    
            
            freeze_layers(net,fe_bool=True,cls_bool=False)
            feat_loss,l_ce,l_dro=0.,0.,0.
         
            for i,(data,labels) in enumerate(trainloader):
                feats,outputs=net(data)
                feat_optim.zero_grad()
                
                for idx,loss_func in enumerate(train_feat_losses):
                    if type(loss_func)==nn.CrossEntropyLoss:
                        l_ce=loss_func(outputs,labels.long().reshape(-1))
                        # print('ce_loss:{}'.format(l))
                    elif type(loss_func)==DROLoss:
                        l_dro=loss_func(feats,labels.reshape(-1),features,targets.reshape(-1))
                        # print('dro_loss:{}'.format(l))
                    feat_loss=(l_ce+l_dro*0.001)
                    # print(l_dro,l_ce,'l_dro l_ce,feaat')
                    feat_optim.zero_grad()
                    feat_loss.backward(retain_graph=True)
                    feat_optim.step()
        else:
            #training classifier
            freeze_layers(net,fe_bool=False,cls_bool=True)
            # for name,params in net.named_parameters():
            #     print(params.requires_grad,name,'cls')
            
            for i,(data,labels) in enumerate(trainloader):
                feats,outputs=net(data)
                for idx,loss_func in enumerate(train_class_losses):
                    if type(loss_func)==nn.CrossEntropyLoss:
                        l_ce=loss_func(outputs,labels.long().reshape(-1))
                    elif type(loss_func)==DROLoss:
                        l_dro=loss_func(feats,labels,features,targets)
                
                    cls_loss+=(l_ce+l_dro*0.001)
                    cls_optim.zero_grad()
                    cls_loss.backward(retain_graph=True)
                    cls_optim.step()
    
        if epoch%20==0:
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,num_classes,device)
            
            f=open(path+'/log_DROLT/log_DROLT.txt','a')
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
    f=open(path+'/log_DROLT/log_DROLT.txt','a')
    f.write(str(['best_metrics:',best_metrics]))      
    f.write('\n')
    return best_metrics
def train(path):
    if os.path.exists(path+'/log_DROLT/log_DROLT.txt'):
        print('Remove log_DROLT.txt')
        os.remove(path+'/log_DROLT/log_DROLT.txt') 
    elif os.path.exists(path+'/log_DROLT'):
        pass
    else:
        os.mkdir(path+'log_DROLT')
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,attr_count,num_classes=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)
    net=Net(attr_count,num_classes)
    
    cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path)

if __name__=='__main__':
    path='../dermatology-5-fold/'
    train(path)
    