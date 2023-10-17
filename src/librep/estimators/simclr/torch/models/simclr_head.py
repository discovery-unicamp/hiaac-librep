import torch
import torch.nn as nn
from librep.estimators.simclr.torch.models.model_base import BaseModel

class SimCLRHead(nn.Module):
    def __init__(self, input_shape,n_components, hidden_1=256, hidden_2=128, hidden_3=50):
        base_model = BaseModel(input_shape,n_components)        
        super(SimCLRHead, self).__init__()
        self.base_model = base_model
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.hidden_3 = hidden_3
        self.projection_1 = nn.Linear(base_model.conv3.out_channels, hidden_1)
        self.relu = nn.ReLU()
        self.projection_2 = nn.Linear(hidden_1, hidden_2)
        self.projection_3 = nn.Linear(hidden_2, hidden_3)

    def forward(self, x):
        base_model_output = self.base_model(x)
        projection_1 = self.projection_1(base_model_output)
        projection_1 = self.relu(projection_1)
        projection_2 = self.projection_2(projection_1)
        projection_2 = self.relu(projection_2)
        projection_3 = self.projection_3(projection_2)
        return projection_3
