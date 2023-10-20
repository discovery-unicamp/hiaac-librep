import torch
import torch.nn as nn

class LinearModel_Intermediate(nn.Module):
    def __init__(self, simclr_head, num_classes):
        super(LinearModel_Intermediate, self).__init__()
        self.base_model = simclr_head.base_model
        self.linear = nn.Linear(self.base_model.conv3.out_channels, num_classes)

    def forward(self, x):
        simclr_output = self.base_model(x)
        linear_output = self.linear(simclr_output)
        return linear_output

    