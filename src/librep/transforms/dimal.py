import torch
import numpy as np
import sklearn.neighbors as nb
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
import librep.estimators.dimal.Functions_ManifoldLearning as FnManifold
from librep.estimators.dimal.MDSNet_3D import MDSNet

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike

class DIMALDimensionalityReduction(Transform):

    def __init__(self, torch_seed=1000, num_landmarks=500, size_HL=70, num_HL=2, n_neighbors=10, latent_dim=10, cuda_device_name=None, force_connection=True):
        self.latent_dim = latent_dim
        self.n_neighbors = n_neighbors
        self.cuda_device_name = cuda_device_name
        self.model = None
        self.num_landmarks = num_landmarks
        self.size_HL = size_HL
        self.num_HL = num_HL
        self.force_connection = force_connection
        torch.manual_seed(torch_seed)
        
    def force_connection_in_graph(self, points, graph):
        n_components, elements = connected_components(graph)
        while n_components > 1:
            all_data = [(elem[0], np.array(elem[1]), elements[elem[0]]) for elem in enumerate(points)]
            groups = []
            for group_id in range(n_components):
                group = [elem for elem in all_data if elem[2] == group_id]
                groups.append(group)
            first_group = groups[0]
            all_other_groups = groups[1:]
            for group in all_other_groups:
                distances = cdist([elem[1] for elem in first_group], [elem[1] for elem in group])
                all_min_distances = np.argwhere(distances==np.min(distances))
                # Pick first
                min_distance_point = all_min_distances[0]
                pointA = first_group[min_distance_point[0]]
                pointB = group[min_distance_point[1]]
                graph[pointA[0], pointB[0]] = np.min(distances)
                graph[pointB[0], pointA[0]] = np.min(distances)
                
            n_components, elements = connected_components(graph)
        return graph
    
    def fit(self, X: ArrayLike, y: ArrayLike = None):
        # Compute the Graph
        W = nb.kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='distance', metric='minkowski',
                                p=2, metric_params=None, include_self=False, n_jobs=-1)

        if self.force_connection:
            W = self.force_connection_in_graph(X, W)
        
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
        MDSNet_3D_result = MDSNet(X, self.Landmarks_FPS, self.NetParams, self.LearningParams, self.latent_dim, cuda_device_name=self.cuda_device_name)
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