import torch
import numpy as np
import sklearn.neighbors as nb
import librep.estimators.dimal.Functions_ManifoldLearning as FnManifold
from librep.estimators.dimal.MDSNet_3D import MDSNet

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike

class DIMALDimensionalityReduction(Transform):

    def __init__(self, torch_seed=1000, num_landmarks=500, size_HL=70, num_HL=2, n_neighbors=25, cuda_device_name=None):
        self.n_neighbors = n_neighbors
        self.cuda_device_name = cuda_device_name
        self.model = None
        self.num_landmarks = num_landmarks
        self.size_HL = size_HL
        self.num_HL = num_HL
        torch.manual_seed(torch_seed)
        
    
    def fit(self, X: ArrayLike, y: ArrayLike = None):
        # Compute the Graph
        W = nb.kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='distance', metric='minkowski',
                                p=2, metric_params=None, include_self=False, n_jobs=-1)

        Landmarks = FnManifold.FPS(W, self.num_landmarks)

        self.numIndices = 200
        self.NetParams = {
            'Size_HL': self.size_HL,
            'Num_HL': self.num_HL
        }
        self.LearningParams = {'Numiter':2000,'learning_rate':1e-2, 'tol':1e-3}
        
        self.Landmarks_FPS = {
            'indices':Landmarks['indices'][0:self.numIndices],
            'Ds':Landmarks['Ds'][0:self.numIndices]
        }
        # Run MDS
        MDSNet_3D_result = MDSNet(X, self.Landmarks_FPS, self.NetParams, self.LearningParams)
        self.model = MDSNet_3D_result[1]

    def transform(self, X: ArrayLike):
        input_HD = torch.from_numpy(X).float()
        if self.cuda_device_name:
            input_HD = input_HD.to(self.cuda_device_name)
        result_LD = self.model(input_HD).detach()
        if self.cuda_device_name:
            result_LD = result_LD.cpu()
        result_LD = result_LD.numpy()
        return result_LD

    def inverse_transform(self, X: ArrayLike):
        self.asd = 0
