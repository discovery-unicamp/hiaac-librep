#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 11:53:47 2017

@author: gipadmin
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myFunctions
from sklearn.neighbors import NearestNeighbors
import sklearn.neighbors as nb
from sklearn import manifold
from sklearn import preprocessing
from scipy import sparse, linalg
from matplotlib import cm
import MDSNet_3D
import Functions_ManifoldLearning as FnManifold
from sklearn import preprocessing, manifold, datasets
import torch
import scipy.io as sio

torch.manual_seed(1000)
#==============================================================================
#                   Generate The Data
#==============================================================================
Numpts = np.power(2,11)
#D, color = datasets.samples_generator.make_s_curve(n_samples=Numpts, noise=0.0, random_state=1)
D, color = datasets.make_s_curve(n_samples=Numpts, random_state=0)

index = np.random.permutation(np.arange(D.shape[0]))
D = D[index,:]
color = color[index]

myFunctions.tic()
# Compute the Graph
W = nb.kneighbors_graph(D, 10, mode='distance',
                              metric='minkowski', p=2, metric_params=None, 
                              include_self=False, n_jobs=-1)
Landmarks = FnManifold.FPS(W,500)
myFunctions.toc()

numIndices = 200
NetParams = {'Size_HL':70,'Num_HL':2}
LearningParams = {'Numiter':2000,'learning_rate':1e-2, 'tol':1e-3}

Landmarks_FPS = {'indices':Landmarks['indices'][0:numIndices],
                     'Ds':Landmarks['Ds'][0:numIndices]}
    
#==============================================================================
#                       RunMDS
#==============================================================================
Embedding_MDSNet,Net,train_err  = MDSNet_3D.MDSNet(D,Landmarks_FPS,NetParams,LearningParams)
myFunctions.toc()
# print(Embedding_MDSNet.shape)
#==============================================================================
#                       Plot
#==============================================================================
plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_aspect('equal')
ax1.scatter(Embedding_MDSNet[:,0],Embedding_MDSNet[:,1], c = color,s=20,lw=0,alpha=1,cmap=cm.jet)

ax1.scatter(Embedding_MDSNet[Landmarks_FPS['indices'],0],Embedding_MDSNet[Landmarks_FPS['indices'],1], c = 'k',s=20,lw=0,alpha=1,cmap=cm.jet)

ax1.set_title(str(numIndices),fontsize = 10)

    
   



















