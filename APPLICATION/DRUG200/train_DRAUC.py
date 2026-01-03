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

def _check_tensor_shape(inputs, shape=(-1, 1)):
    input_shape = inputs.shape
    target_shape = shape
    if len(input_shape) != len(target_shape):
        inputs = inputs.reshape(target_shape)
    return inputs
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class DRAUCOptim(torch.optim.Optimizer):
    """
    Modified from PESG in LibAUC
    """
    def __init__(self, 
                 model, 
                 loss_fn=None,
                 a=None,          # to be deprecated
                 b=None,          # to be deprecated
                 alpha=None,      # to be deprecated 
                 margin=1.0, 
                 lr=0.1, 
                 gamma=None,      # to be deprecated 
                 clip_value=1.0, 
                 weight_decay=1e-5, 
                 epoch_decay=2e-3, # default: gamma=500
                 momentum=0, 
                 verbose=True,
                 device=None,
                 **kwargs):

        if not device:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:  
            self.device = device      
        assert (gamma is None) or (epoch_decay is None), 'You can only use one of gamma and epoch_decay!'
        if gamma is not None:
           assert gamma > 0
           epoch_decay = 1/gamma
        
        self.margin = margin
        self.model = model
        self.lr = lr
        self.gamma = gamma  # to be deprecated
        self.clip_value = clip_value
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epoch_decay = epoch_decay
               
        self.loss_fn = loss_fn
        if loss_fn != None:
            try:
                self.a = loss_fn.a 
                self.b = loss_fn.b 
                self.alpha = loss_fn.alpha 
            except:
                print('AUCLoss is not found!')
        else:
            self.a = a 
            self.b = b 
            self.alpha = alpha
            # self._lambda = _lambda           
        # print(self.model)
        self.model_ref = self.init_model_ref()
        
        self.model_acc = self.init_model_acc()
        self.T = 0                # for epoch_decay
        self.steps = 0            # total optim steps
        self.verbose = verbose    # print updates for lr/regularizer
    
        def get_parameters(params):
            for p in params:
                yield p
        if self.a is not None and self.b is not None:
           self.params = get_parameters(list(model.parameters())+[self.a, self.b])#, self._lambda])
        else:
           self.params = get_parameters(list(model.parameters()))
        
        # self.fix_lambda = fix_lambda

        self.defaults = dict(lr=self.lr, 
                             margin=margin, 
                             a=self.a, 
                             b=self.b,
                             alpha=self.alpha,
                             clip_value=clip_value,
                             momentum=momentum,
                             weight_decay=weight_decay,
                             epoch_decay=epoch_decay,
                             model_ref=self.model_ref,
                             model_acc=self.model_acc
                             )
        
        super(DRAUCOptim, self).__init__(self.params, self.defaults)
         
    def __setstate__(self, state):
        super(DRAUCOptim, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def init_model_ref(self):
        self.model_ref = []
        for var in list(self.model.parameters())+[self.a, self.b]:#, self._lambda]: 
            if var is not None:
                self.model_ref.append(torch.empty(var.shape).normal_(mean=0, std=0.01).to(self.device))
        return self.model_ref
     
    def init_model_acc(self):
        self.model_acc = []
        for var in list(self.model.parameters())+[self.a, self.b]:#, self._lambda]: 
            if var is not None:
               self.model_acc.append(torch.zeros(var.shape, dtype=torch.float32,  device=self.device, requires_grad=False).to(self.device)) 
        return self.model_acc
    
    @property    
    def optim_steps(self):
        return self.steps
    
    @property
    def get_params(self):
        return list(self.model.parameters())
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            clip_value = group['clip_value']
            momentum = group['momentum']
            self.lr =  group['lr']
            
            epoch_decay = group['epoch_decay']
            model_ref = group['model_ref']
            model_acc = group['model_acc']
            
            m = group['margin'] 
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            
            # updates
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = torch.clamp(p.grad.data , -clip_value, clip_value) + epoch_decay*(p.data - model_ref[i].data) + weight_decay*p.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(1-momentum).add_(d_p, alpha=momentum)
                    d_p =  buf
                p.data = p.data - group['lr']*d_p
                model_acc[i].data = model_acc[i].data + p.data
            
            if alpha is not None: 
               if alpha.grad is not None: 
                  alpha.data = alpha.data + group['lr']*(2*(m + b.data - a.data)-2*alpha.data)
                  alpha.data  = torch.clamp(alpha.data,  0, 999)

        self.T += 1  
        self.steps += 1
        return loss

    def zero_grad(self):
        self.model.zero_grad()
        if self.a is not None and self.b is not None:
           self.a.grad = None
           self.b.grad = None
        if self.alpha is not None:
           self.alpha.grad = None
        
    def update_lr(self, decay_factor=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            if self.verbose:
               print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))
            
    def update_regularizer(self, decay_factor=None):
        if self.verbose:    
           print ('Updating regularizer @ T=%s!'%(self.steps))
        for i, param in enumerate(self.model_ref):
            self.model_ref[i].data = self.model_acc[i].data/self.T
        for i, param in enumerate(self.model_acc):
            self.model_acc[i].data = torch.zeros(param.shape, dtype=torch.float32, device=self.device,  requires_grad=False).to(self.device)
        self.T = 0
        
class DRAUCLoss(torch.nn.Module): # AUC margin loss with extra k to gurantee convexity of alpha
    def __init__(self, margin=1.0, k = 1, _lambda = 1., device=None):
        super(DRAUCLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.k = k
        assert self.k > 0
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device) 
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self._lambda = torch.tensor(_lambda, dtype=torch.float32, device=self.device, requires_grad=False).to(self.device)

    def mean(self, tensor):
        #return torch.sum(tensor)/(torch.count_nonzero(tensor) + 1e-6)
        return torch.mean(tensor)

    def stop_grad(self):
        self.a.requires_grad = False
        self.b.requires_grad = False
        self.alpha.requires_grad = False
    
    def start_grad(self):
        self.a.requires_grad = True
        self.b.requires_grad = True
        self.alpha.requires_grad = True

    def forward(self, y_pred, y_true):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        pos_mask = (1==y_true).float()
        neg_mask = (0==y_true).float()
        #print(self.alpha,'alpha')
        loss = self.mean((y_pred - self.a)**2*pos_mask) + \
               self.mean((y_pred - self.b)**2*neg_mask) + \
               2*self.alpha*(self.margin + self.mean(y_pred*neg_mask) - self.mean(y_pred*pos_mask)) - \
               self.k * self.alpha**2
        if torch.isnan(loss):
            raise ValueError            
        return loss   
def attack_DRO(model, f, x, y, _lambda, attack_lr, \
                 epsilon = 0.08, iters = 10, projection = False, p = np.inf, constrained = False, _lambda1 = None, epsilon1 = None):
    '''
    model: Classifier
    f: Loss function, e.g., AUC Loss
    x: Original input image
    y: Label
    _lambda: Regularization parmeter lambda
    attack_lr: Learning rate for gradient ascent
    epsilon: maximum attack distance
    iters: Iteration nums for gradient ascent for maximizaiton
    projection: Weather to restrict delta in epsilon Lp norm ball
    p: Use Lp norm to evaluate the distance
    '''
    if isinstance(f, DRAUCLoss):
        f.stop_grad()

    model.eval()
    # max_loss = torch.zeros(y.shape[0]).cuda(0)
    # max_delta = torch.zeros_like(x).cuda(0)
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(x)
    
    # delta = torch.zeros_like(x).cuda(0) #init delta
    delta = torch.zeros_like(x) #init delta
    delta.normal_()

    if constrained:
        y = y.squeeze(dim = -1)
        assert _lambda1 is not None and epsilon1 is not None
        # print(delta[y==1].shape, delta[y==0].shape)
        if p==2: # restrict adv samples in epsilon p-norm ball
            d_flat = delta[y==0].view(delta[y==0].size(0),-1)
            d_flat1 = delta[y==1].view(delta[y==1].size(0),-1)
            n = d_flat.norm(p=p,dim=1).view(delta[y==0].size(0),1,1,1)
            n1 = d_flat1.norm(p=p,dim=1).view(delta[y==1].size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            r1 = torch.zeros_like(n1).uniform_(0, 1)
            delta[y==1] *= r1/n1*epsilon1
            delta[y==0] *= r/n*epsilon
        elif p == np.inf:
            delta[y==0].uniform_(-epsilon, epsilon)
            delta[y==1].uniform_(-epsilon1, epsilon1)
        else:
            raise ValueError

        _lambda = y * _lambda1 + (1-y) * _lambda
        y = y.unsqueeze(dim = -1)
    else:
        if p==2: # restrict adv samples in epsilon p-norm ball
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=p,dim=1).view(delta.size(0),1)
            r = torch.zeros_like(n).uniform_(0, 1)
            
            delta *= r/n*epsilon
            
        elif p == np.inf:
            delta.uniform_(-epsilon, epsilon)
        else:
            raise ValueError

    delta = clamp(delta, -x, 1-x)
    delta.requires_grad = True
    
    for _ in range(iters): # generate DRO adv samples
        # loss = f(torch.sigmoid(model(normalize(x+delta))), y) - (_lambda * torch.pow(torch.norm(delta, p=p),2)).mean()
        loss = f(torch.sigmoid(model(x+delta)), y) - (_lambda * torch.pow(torch.norm(delta, p=p),2)).mean()
        loss.backward()

        grad = delta.grad.detach()
        
        d = delta
        if p == 2:
            g_norm = torch.norm(grad.view(grad.shape[0],-1),dim=1).view(-1,1)
            scaled_g = grad/(g_norm + 1e-10)
            # print(g_norm.shape,scaled_g.shape,'g_norm')
            if projection:
                d = (d + scaled_g * attack_lr).view(delta.size(0),-1).renorm(p=p,dim=0,maxnorm=epsilon).view_as(delta)
            else:
                d = d + scaled_g * attack_lr
        elif p==np.inf:
            d = d + attack_lr * torch.sign(grad)
        # print(delta.shape,d.shape,'delta')
        d = clamp(d, -x, 1-x)
        delta.data = d
        delta.grad.zero_()
        
        # all_loss = F.binary_cross_entropy(torch.sigmoid(model(normalize(x+delta))), y, reduction='none').squeeze()
        # print(x.shape,delta.shape)
        all_loss = F.binary_cross_entropy(torch.sigmoid(model(x+delta)), y, reduction='mean').squeeze()
        #print(all_loss,'all_loss',all_loss.shape,max_loss.shape)
        #print(all_loss,'all_loss',(all_loss >= max_loss).shape,delta.shape)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
        #break
    model.train()
    if isinstance(f, DRAUCLoss):
        f.start_grad()

    return max_delta

def cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path):
    init_net=copy.deepcopy(net)
    best_metrics_set=[]
    for i in range(len(cv_trainloader)):
        print('%d fold cross validation:'%(i+1))
        best_metrics=train_drauc(net,cv_trainloader[i],cv_testloader[i],num_classes,path)
        best_metrics_set.append(best_metrics)
        net=init_net
    
    f=open(path+'result.txt','a')
    print('5-cross validation result:',np.array(best_metrics_set).mean(axis=0))
    f.write(str(['DRAUC',np.array(best_metrics_set).mean(axis=0)]))
    f.write('\n')
    f.close()
def test(net,testloader,num_classes,device):
    macro_avg_precision_set,macro_avg_recall_set,macro_avg_f1_set,_matthews_corrcoef_set,macro_avg_roc_auc_score_set=[],[],[],[],[]
    for _,data in enumerate(testloader):
        inputs,labels=data
        inputs,labels=inputs.to(device).float(),labels.to(device)
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
        print(np.unique(labels))
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
def train_drauc(net,trainloader,testloader,num_classes,path):
    lr=0.01
    eps=8
    lambda_lr=0.01
    attack_lr=15
    epsilon=128
    attack_lr /= 255.
    epsilon /= 255.
    margin=1.
    lambda_=1.
    momentum=0.9
    wd=5e-4
    warmup_epochs=0
    lambda_grad=0.004
    attack_iters=10
    projection=False
    norm=2
    
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    net.to(device)
    EPOCH=1000
    
    criterion = DRAUCLoss(margin = margin, k = 1., _lambda = lambda_)
    optimizer = DRAUCOptim(
        net, 
        a = criterion.a, 
        b = criterion.b, 
        alpha = criterion.alpha,
        lr = lr,
        constrained=True,
        momentum=momentum,
        weight_decay = wd, 
        epoch_to_opt=warmup_epochs,
        
    )
    # optimizer=torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    
    lambda_grad = torch.tensor(lambda_grad, requires_grad = True)
    
    
    # weights=torch.zeros(num_classes)
    # for _,data in enumerate(trainloader):
    #     inputs,labels=data
        
    #     for i in range(labels.shape[0]):
    #         weights[int(labels[i])]+=1
    
    metrics,best_metrics=-10,[]
    # weights=(1./(weights+1e-8)).to(device)
    # loss_func=nn.CrossEntropyLoss(weight=weights)
    
    for epoch in range(EPOCH):
        for _,data in enumerate(trainloader):
            # print(net,'net',trainloader)
            inputs,labels=data
            
            inputs,labels=inputs.to(device),labels.to(device).long()
            labels_one_hot=F.one_hot(labels.long(),num_classes).reshape(-1,num_classes).float()
            # labels=labels.reshape(inputs.shape[0],-1)
            # print(inputs.shape)
            delta = attack_DRO(net, criterion, inputs, labels_one_hot, criterion._lambda, attack_lr, epsilon , attack_iters, projection, norm)
            inputs = torch.clamp(inputs + delta[:inputs.size(0)], 0, 1)
            #plt.imshow(inputs[0].reshape(28,28).numpy())
            #plt.pause(0.01)
            # print(inputs.shape,'inputs shape')
            outputs = net(inputs)
            loss = criterion(outputs, labels_one_hot)

            
            # print(labels[0],labels_one_hot[0])
            # outputs=net(inputs)
            
            # loss=loss_func(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(loss,'loss')
            # break
        if epoch%20==0:
            print('start test...')
            precision,recall,f1,mcc,auc=test(net,testloader,num_classes,device)
            
            f=open(path+'/log_DRAUC/log_DRAUC.txt','a')
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
    f=open(path+'/log_DRAUC/log_DRAUC.txt','a')
    f.write(str(['best_metrics:',best_metrics]))      
    f.write('\n')
    return best_metrics
def train(path):
    if os.path.exists(path+'/log_DRAUC/log_DRAUC.txt'):
        print('Remove log_DRAUC.txt')
        os.remove(path+'/log_DRAUC/log_DRAUC.txt') 
    elif os.path.exists(path+'/log_DRAUC'):
        pass
    else:
        os.mkdir(path+'log_DRAUC')
    
    
    # train_data_all,test_data_all=rd.read_data(path)
    #print(train_data_all,test_data_all)
    cv_trainloader,cv_testloader,attr_count,num_classes=rd.construct_cv_trainloader()
    
    torch.manual_seed(0)
    net=Net(attr_count,num_classes)
    
    cv_train_au(net,cv_trainloader,cv_testloader,attr_count,num_classes,path)

if __name__=='__main__':
    path='../drug200/'
    train(path)
    
