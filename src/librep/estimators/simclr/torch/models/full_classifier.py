import torch
import torch.nn as nn

class FullClassifier(nn.Module):
    def __init__(self, base_model, intermediate_dim, output_dim):
        super(FullClassifier, self).__init__()
        self.base_model = base_model
        self.fc1 = nn.Linear(intermediate_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        base_output = self.base_model(x)
        x = self.relu(self.fc1(base_output))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
