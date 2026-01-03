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
import read_data as rd

class Net(nn.Module):
    def __init__(self,attr_count,num_classes):
        super(Net,self).__init__()
        self.layer0=nn.Sequential(
            nn.Linear(attr_count,128),
            nn.ReLU(),
            )
        self.layer1=nn.Sequential(
            nn.Linear(128,num_classes),
            nn.ReLU()
            )
    def forward(self,x):
        return self.layer1(self.layer0(x.view(x.size(0),-1)))

    
class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M

        self.LWS = LWS
        self.reg_ETF = reg_ETF
#        if LWS:
#            self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
#            self.alpha = nn.Parameter(1e-3 * torch.randn(1, num_classes).cuda())
#            self.learned_norm = (F.softmax(self.alpha, dim=-1) * num_classes)
#        else:
#            self.learned_norm = torch.ones(1, num_classes).cuda()

        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`

    """

    def __init__(self, eps, max_iter, dis, gpu, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.dis = dis
        self.gpu = gpu

    def forward(self, x, y):
        
        d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if self.dis == 'cos':
            C = 1-d_cosine(x_col, y_lin)
        elif self.dis == 'euc':
            C= torch.mean((torch.abs(x_col - y_lin)) ** 2, -1)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        actual_nits = 0
        thresh = 1e-1

        for i in range(self.max_iter):
            u1 = u  
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    def _cost_matrix(x, y, dis, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if dis == 'cos':
            C = 1-d_cosine(x_col, y_lin)
        elif dis == 'euc':
            C= torch.mean((torch.abs(x_col - y_lin)) ** p, -1)

        return C



    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def cv_train_au(net,classifier_etf,cv_trainloader,cv_testloader,attr_count,num_classes,path):
    init_net=copy.deepcopy(net)
    best_metrics_set=[]
    for i in range(len(cv_trainloader)):
        print('%d fold cross validation:'%(i+1))
        best_metrics=train_disa(net,classifier_etf,cv_trainloader[i],cv_testloader[i],attr_count,i+1,num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['DISA',np.array(best_metrics_set).mean(axis=0)]))
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
def train_disa(net,classifier_etf,trainloader,testloader,attr_count,fold,num_classes,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    net.to(device)
    classifier_etf.to(device)
    sinkhorna_ot = SinkhornDistance(eps=1, max_iter=200, reduction=None, dis='cos', gpu=0)

    EPOCH=1000
    optimizer=torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    weights=torch.zeros(num_classes)
    for _,data in enumerate(trainloader):
        inputs,labels=data
        
        for i in range(labels.shape[0]):
            weights[int(labels[i])]+=1
    
    metrics,best_metrics=0,[]
    weights=(1./(weights+1e-8)).to(device)
    loss_func=nn.CrossEntropyLoss()
    metrics,best_metrics=0,[]
    for epoch in range(EPOCH):
        for _,data in enumerate(trainloader):
            inputs,labels=data
            
            inputs,labels=inputs.to(device),labels.to(device).long()
            
            batch_size = inputs.size()[0]
            
            index = torch.randperm(batch_size).to(device)
            #lam = np.random.beta(args.alpha, args.alpha)
            lam = np.random.beta(0.1, 0.1)
            mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
            
            y_a, y_b = labels, labels[index]
            
            #labels_one_hot=F.one_hot(labels.long(),num_classes).reshape(-1,num_classes).long()
            # print(labels[0],labels_one_hot[0])
            outputs=net(mixed_x)
            #print(inputs.shape,'outputs')
            feature=torch.zeros(inputs.shape[0],128).to(device)
            for name,module in net.named_children():
                feature=module(mixed_x.to(device))
                # print(feature.shape,'feature')
                if 'layer0' in name:
                    feature=feature.data.reshape(inputs.shape[0],128)
                    break
                mixed_x=feature
            
            feat_etf=classifier_etf(feature)    
            # print(feat_etf.shape,'fea',y_a,y_b)
            y_a,y_b=y_a.reshape(-1),y_b.reshape(-1)
            loss_ot = sinkhorna_ot(feat_etf, classifier_etf.ori_M.T)
            loss = loss_func(outputs, y_a) * lam + loss_func(outputs, y_b) * (1. - lam) + 0.1 * loss_ot
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(loss,'loss')
            # break
        if epoch%20==0:
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,num_classes,device)
            
            f=open(path+'/log_DISA/log_DISA.txt','a')
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
    f=open(path+'/log_DISA/log_DISA.txt','a')
    f.write(str(['best_metrics:',best_metrics]))      
    f.write('\n')
    return best_metrics
def train(path):
    if os.path.exists(path+'/log_DISA/log_DISA.txt'):
        print('Remove log_DISA.txt')
        os.remove(path+'/log_DISA/log_DISA.txt') 
    elif os.path.exists(path+'/log_DISA'):
        pass
    else:
        os.mkdir(path+'log_DISA')
    
    train_data_all,test_data_all=rd.read_data(path)
    cv_trainloader,cv_testloader,attr_count,num_classes=rd.construct_cv_trainloader(train_data_all,test_data_all)
    
    torch.manual_seed(0)

    net=Net(attr_count,num_classes)
    # print(net)
    classifier_etf=ETF_Classifier(128,num_classes)
    
    cv_train_au(net,classifier_etf,cv_trainloader,cv_testloader,attr_count,num_classes,path)

if __name__=='__main__':
    path='../shuttle-5-fold/'
    train(path)
    
