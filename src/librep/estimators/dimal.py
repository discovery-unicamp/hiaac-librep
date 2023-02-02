
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DIMAL(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i-1], hidden_dims[i])
                                            for i in range(1, len(hidden_dims))])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x
    
    def loss(self, outputs, targets):
        return F.mse_loss(outputs, targets)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        # loss = self.loss(outputs, targets)
        # return {'loss': loss}
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)
