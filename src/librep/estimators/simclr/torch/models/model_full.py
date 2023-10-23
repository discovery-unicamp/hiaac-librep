import torch.nn as nn
       
class Model_Full(nn.Module):
    def __init__(self, simclr_head, num_classes):
        super(Model_Full, self).__init__()
        self.simclr_head = simclr_head
        self.linear = nn.Linear(simclr_head.hidden_3, num_classes)

    def forward(self, x):
        simclr_output = self.simclr_head(x)
        linear_output = self.linear(simclr_output)
        return linear_output

    
