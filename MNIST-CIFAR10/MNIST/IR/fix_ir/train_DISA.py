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

class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda()

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
        
        d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8).cuda(self.gpu)
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        if self.dis == 'cos':
            C = 1-d_cosine(x_col.cuda(self.gpu), y_lin.cuda(self.gpu))
        elif self.dis == 'euc':
            C= torch.mean((torch.abs(x_col - y_lin)) ** 2, -1)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).cuda(self.gpu).squeeze()
        
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).cuda(self.gpu).squeeze()

        u = torch.zeros_like(mu).cuda(self.gpu)
        v = torch.zeros_like(nu).cuda(self.gpu)

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
    f.write(str(['CE',np.array(best_metrics_set).mean(axis=0)]))
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
def train_disa(net,classifier_etf,trainloader,testloader,num_classes,path):
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    net.to(device)
    classifier_etf.to(device)
    sinkhorna_ot = SinkhornDistance(eps=1, max_iter=200, reduction=None, dis='cos', gpu=0).cuda(0)

    EPOCH=1000
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    weights=torch.zeros(num_classes)
    for _,data in enumerate(trainloader):
        inputs,labels=data
        
        for i in range(labels.shape[0]):
            weights[int(labels[i])]+=1
    
    metrics,best_metrics=0,[]
    weights=(1./(weights+1e-8)).to(device)
    loss_func=nn.CrossEntropyLoss(weight=weights)
    metrics,best_metrics=0,[]
    for epoch in tqdm(range(EPOCH),ncols=80):
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
            #print(outputs,'outputs')
            feature=torch.zeros(inputs.shape[0],512).to(device)
            for name,module in net.named_children():
                feature=module(mixed_x.to(device))
                if 'avgpool' in name:
                    feature=feature.data.reshape(inputs.shape[0],net.fc.in_features)
                    break
                mixed_x=feature
            
            feat_etf=classifier_etf(feature)    
            
            loss_ot = sinkhorna_ot(feat_etf, classifier_etf.ori_M.T)
            loss = loss_func(outputs, y_a) * lam + loss_func(outputs, y_b) * (1. - lam) + 0.1 * loss_ot
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(loss,'loss')
            # break
        if epoch%20==0:
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,device)
            
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
    net.avgpool=nn.AdaptiveAvgPool2d(1) #model
    
    num_features=net.fc.in_features
    
    net.fc=nn.Linear(num_features,10)#classifier
    classifier_etf=ETF_Classifier(num_features,10)
    
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
                                batch_size=200,
                                shuffle=True)
    train_disa(net,classifier_etf,trainloader,testloader,10,path)
    

if __name__=='__main__':
    path='./'
    train(path)
    
