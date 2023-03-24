#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 00:24:24 2017
@author: gipadmin
"""
import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable


def MDSNet(Data,Landmarks,NetParams,LearningParams):
#----------------------- Build the Batches ------------------------------------
    DataV = Variable(torch.FloatTensor(Data))  
    Ds = Landmarks['Ds'];
    indices_FPS = Landmarks['indices']
    
    Dsk = Ds[:,indices_FPS]
    numFPS = Dsk.shape[0]    
    batch = np.zeros([int(numFPS*(numFPS-1)/2),3])
    # print('batch', batch.shape, 'numFPS', numFPS)
    index = 0    
    for i in range(0,numFPS):
        for j in range(i+1,numFPS):
            batch[index,:] = [i,j,Dsk[i,j]]
            index = index+1
    
    D = Variable(torch.FloatTensor(Data[indices_FPS,:]))
    d = Variable(torch.FloatTensor(batch[:,2]))
    sumd = d.pow(2).sum()
    # print(Ds)
#==============================================================================
#                   Define the Architecture
#==============================================================================
    #torch.manual_seed(666)#1000 was the typical
    HiddenLayer = NetParams['Size_HL']
    NumberOfHiddenLayers = NetParams['Num_HL']
    
    Net = nn.Sequential()
    Net.add_module('Layer1',nn.Linear(180,HiddenLayer,bias = False))
    Net.add_module('NonLinear1',torch.nn.PReLU(num_parameters=1, init=0.25))

    for layerIter in np.arange(2,NumberOfHiddenLayers+1):
        Net.add_module('Layer'+str(layerIter),nn.Linear(HiddenLayer,HiddenLayer,bias = False))
        Net.add_module('NonLinear'+str(layerIter),torch.nn.PReLU(num_parameters=1, init=0.25))
 
    Net.add_module('LayerEnd',nn.Linear(HiddenLayer,10,bias = False))

    Numiter=LearningParams['Numiter']
    learning_rate=LearningParams['learning_rate']
    tol = LearningParams['tol']
    
    optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate,  betas=(0.96, 0.999))
    #optimizer =torch.optim.Adadelta(Net.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
    
    
    def adjust_lr(optimizer, epoch, learning_rate):
        lr = learning_rate * (0.2 ** (epoch //1500))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
   
##==============================================================================
##                          Train 
##==============================================================================
    train_err = np.zeros(Numiter)
    ind1 =Variable(torch.LongTensor(batch[:,0].astype(int)))
    ind2 =Variable(torch.LongTensor(batch[:,1].astype(int)))
    
    
    for itern in range(0,Numiter):
        # print(itern)
        Y = Net.forward(D)
        # print(itern, Y)
        Y1 = Y.index_select(0,ind1)
        Y2 = Y.index_select(0,ind2)
        
        loss = torch.sum(((Y1-Y2)*(Y1-Y2)), dim=1).pow(0.5).sub(d).pow(2).sum().div(sumd)
        print('LOSS', itern, loss)
        # print(itern) #, torch.isnan(loss).any(), torch.numel(sumd) - torch.count_nonzero(sumd))
        # print(sumd)
        # loss = loss
        
        
        train_err[itern] = loss.data
        
        adjust_lr(optimizer, itern, learning_rate);            
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
##==============================================================================
       
    embedding_MDSNet = Net.forward(DataV).squeeze().data.numpy()
    return embedding_MDSNet,Net,train_err
        
        
