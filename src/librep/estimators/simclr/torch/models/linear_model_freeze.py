import torch.nn as nn
       
class LinearModel_IntermediateFreeze(nn.Module):
    def __init__(self, simclr_head, num_classes):
        super(LinearModel_Intermediate, self).__init__()
        self.base_model = simclr_head.base_model
        self.linear = nn.Linear(self.base_model.conv3.out_channels, num_classes)

        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        simclr_output = self.base_model(x)
        linear_output = self.linear(simclr_output)
        return linear_output

    
