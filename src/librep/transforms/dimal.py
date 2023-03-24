import torch
import sklearn.neighbors as nb
import librep.estimators.dimal.Functions_ManifoldLearning as FnManifold
from librep.estimators.dimal.MDSNet_3D import MDSNet

# import numpy as np

# from librep.estimators.ae.torch.models.topological_ae.topological_ae import (
#     TopologicallyRegularizedAutoencoder
# )
# from tqdm.notebook import tqdm
# from sklearn.model_selection import train_test_split
from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike
# from torch.optim import Adam
# import matplotlib.pyplot as plt
# import pickle

class DIMALDimensionalityReduction(Transform):

    def __init__(self, torch_seed=1000, num_landmarks=500, size_HL=70, num_HL=2):
        self.model = None
        self.num_landmarks = num_landmarks
        self.size_HL = size_HL
        self.num_HL = num_HL
        torch.manual_seed(torch_seed)
        
    
    def fit(self, X: ArrayLike, y: ArrayLike = None):
        # Compute the Graph
        W = nb.kneighbors_graph(X, 10, mode='distance', metric='minkowski',
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
        result_LD = self.model(input_HD).detach().numpy()
        return result_LD

    def inverse_transform(self, X: ArrayLike):
        self.asd = 0