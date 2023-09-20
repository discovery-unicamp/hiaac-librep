import torch.nn as nn


class LinearModel3(nn.Module):
    def __init__(self, base_model, num_classes, intermediate_layer=7):
        super().__init__()
        self.base_model = base_model
        self.output_shape = num_classes
        self.intermediate_layer = intermediate_layer

        self.model = nn.Sequential(
            self.base_model,
            nn.Linear(self.base_model.output_shape[1], num_classes).to(self.base_model.device)
        )

        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x
        
class LinearModel(nn.Module):
    def __init__(self, simclr_head, num_classes):
        super(LinearModel, self).__init__()
        self.simclr_head = simclr_head
        self.linear = nn.Linear(simclr_head.hidden_3, num_classes)

    def forward(self, x):
        simclr_output = self.simclr_head(x)
        linear_output = self.linear(simclr_output)
        return linear_output

    
class LinearModel1(nn.Module):
    def __init__(self,simclr_head, num_classes):
        super(LinearModel, self).__init__()
        self.global_max_pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.projection_1 = nn.Linear(96, 256)  # Certifique-se de ajustar os tamanhos de entrada e saída conforme necessário
        self.relu = nn.ReLU()
        self.projection_2 = nn.Linear(256, 128)  # Certifique-se de ajustar os tamanhos de entrada e saída conforme necessário
        self.projection_3 = nn.Linear(128, 7)  # Certifique-se de ajustar os tamanhos de entrada e saída conforme necessário

    def forward(self, x):
        x = self.global_max_pooling(x)
        x = x.view(x.size(0), -1)  # Transforma a saída em um vetor
        x = self.projection_1(x)
        x = self.relu(x)
        x = self.projection_2(x)
        x = self.projection_3(x)
        return x
    
class LinearModelXX(nn.Module):
    def __init__(self, simclr_head, num_classes):
        super(LinearModel, self).__init__()
        #self.simclr_head = simclr_head
        self.base_model_until_conv3 = nn.Sequential(
            simclr_head.base_model.conv1,
            simclr_head.base_model.dropout1,
            simclr_head.base_model.conv2,
            simclr_head.base_model.dropout2,
            simclr_head.base_model.conv3,
           simclr_head.base_model.dropout3
)
        self.linear = nn.Linear(simclr_head.base_model.conv3.out_channels, num_classes)

    def forward(self, x):
        simclr_output = self.base_model_until_conv3(x)
        linear_output = self.linear(simclr_output)
        return linear_output

    
