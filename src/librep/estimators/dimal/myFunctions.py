#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 12:20:50 2017

@author: gipadmin
"""
import numpy as np
import math
import torch
from torch.autograd import Variable
import time
from scipy import sparse, misc

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from matplotlib import cm

#==============================================================================
#                      Populate the Distance Triplets 
#==============================================================================

def createBatch(D,W, pc_FPS):
    print('creating batch . . .')    
    index = 0
    Numpts = W.shape[0]
#-------------------------------FPS--------------------------------------------
    indices_FPS = [];
    indices_FPS.append(np.random.randint(0,Numpts))
    
    if pc_FPS!=1:
            
        if pc_FPS<1:
           Nsampled = np.int(pc_FPS*Numpts)
        else:
           Nsampled = pc_FPS
           
        Ds = np.zeros([Nsampled,Numpts])
            
        for itern in range(0,Nsampled-1):
            # first compute the distances:
            Ds[itern,:] = sparse.csgraph.dijkstra(W,directed = False,indices=indices_FPS[itern])
            indices_FPS.append(np.argmax(np.ndarray.min(Ds[0:itern+1,:],0)))
            #print(itern)
        Ds[itern+1,:]   = sparse.csgraph.dijkstra(W,directed = False,indices=indices_FPS[-1])
# -----------------------------------------------------------------------------
        Ds_FPS = Ds[:,indices_FPS]
        D = D[indices_FPS,:]
    else:
        Ds_FPS = sparse.csgraph.dijkstra(W,directed = False)
        indices_FPS = np.arange(0,Numpts)                
    
    Numpts_s = Ds_FPS.shape[0]
    
    AllOfThem = np.zeros([(Numpts_s*(Numpts_s-1)/2),3])   
        
    for i in range(0,Numpts_s):
            
        for j in range(i+1,Numpts_s):
                
            AllOfThem[index,:] = [i,j,Ds_FPS[i,j]]
            index = index+1
    
    print('batch created . . .')             
    return D, Ds_FPS, AllOfThem,indices_FPS
   
#==============================================================================
#                      Tic-Toc Function 
#==============================================================================
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)        


#==============================================================================
#                      Conformal Fishbowl Generator
#==============================================================================
def generate_Fishbowl(NPts,R,a):
    
    rc = 1;
    theta = 2*np.pi*np.random.rand(NPts,1);
    r = np.sqrt(np.random.rand(NPts,1));
    alpha = np.multiply(rc*r,np.cos(theta));
    beta =  np.multiply(rc*r,np.sin(theta));
    
    b = R - np.sqrt(R**2-a**2);
    t = np.divide(2*b*R,(alpha**2 + beta**2 + b**2));
    X = t*alpha; Y = t*beta; Z = (1-t)*b;

    Data = np.concatenate((X,Y,Z),axis=1)
    colors = Z.squeeze()
    
    TrueCord = np.hstack((alpha,beta));
    
    return Data,colors,TrueCord
        
#==============================================================================
#                 Helical Ribbon Generator 
#==============================================================================
def generate_tangentDevelopable(powerOfTwo,numberOfRots):
    
    N = np.power(2,powerOfTwo); # number of points considered
    Data = np.zeros([N,3])
    
    E = numberOfRots*np.pi ;a =1.5;

    t = np.random.uniform(0,E,N)
    s = np.random.uniform(-a,a,N)

#    gX = -np.sin(t)/np.sqrt(2);
#    gY = np.cos(t)/np.sqrt(2);
#    gZ = 1/np.sqrt(2);

#    gX = -np.cos(t)/np.sqrt(2);
#    gY = np.sin(t)/np.sqrt(2);
#    gZ = 0;

    Data[:,0] = np.cos(t)
    Data[:,1] = np.sin(t)
    Data[:,2] = t+s
    
    color = t
        
    return Data, color

#==============================================================================
#                      Point Plotter
#==============================================================================
def plotPoints(D,color,pc,fig,dim):

    if pc<1.1:
       numpts = np.int(pc*D.shape[0])
    else:
       numpts = pc
       
    index = np.random.permutation(np.arange(D.shape[0]))
    index = index[0:numpts]
    D = D[index,:]
    color = color[index]   
    
    if dim==3:
       ax = fig.add_subplot(111,projection="3d")
       ax.scatter(D[:,0],D[:,1],D[:,2],c = color,s=10,lw=0,alpha=1,cmap=cm.jet)
    else:
       ax = fig.add_subplot(111)
       ax.set_aspect(1)
       ax.scatter(D[:,0],D[:,1],c = color,s=5,lw=0,alpha=1,cmap=cm.jet)

###############################################################################
#               Mixture Of Sinusoid Manifold Generator
###############################################################################
def imageSinusoiManifoldGenerator(ImSize,NumberOfSamples,omega1,omega2,mix):
    
    heightParam = 5;
    
    tx = np.linspace(0,2*np.pi,ImSize)
    ty = np.linspace(0,heightParam,ImSize)
    
    X,_ = np.meshgrid(tx,tx)
    _,Y = np.meshgrid(ty,ty)
    Y = np.matlib.repmat(Y.reshape(1,(ImSize*ImSize)),NumberOfSamples,1)
    
    
    sin1 = np.sin(omega1*X).reshape(1,(ImSize*ImSize))
    sin2 = np.sin(omega2*X).reshape(1,(ImSize*ImSize))
    
    sin = np.concatenate((sin1,sin2),axis = 0)
    Mixtures = np.random.uniform(mix[0],mix[1],[NumberOfSamples,2])
    
    Data = np.less(Y,heightParam/2+np.matmul(Mixtures,sin)).astype(float)
    
    colors = np.sqrt(np.sum(np.multiply(Mixtures,Mixtures),1))
#    colors = Mixtures[:,1]
    
    y = np.ascontiguousarray(Data).view(np.dtype((np.void, Data.dtype.itemsize * Data.shape[1])))
    _, idx = np.unique(y, return_index=True)

    Data = Data[idx,:]
    Mixtures = Mixtures[idx,:]
    colors = colors[idx]
    
    return Data, Mixtures, colors
#------------------------------------------------------------------------------



            
            
            
        