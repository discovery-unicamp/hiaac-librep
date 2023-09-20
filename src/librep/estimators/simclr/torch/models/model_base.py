import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_shape, model_name="base_model"):
        super(BaseModel, self).__init__()
        self.input_shape = input_shape
        self.model_name = model_name
        self.build()

    def build(self):
        self.conv1 = nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=24)
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16)
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=8)
        self.dropout3 = nn.Dropout(p=0.1)
        self.global_max_pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to (batch_size, num_channels, time_steps)
        x = self.conv1(x)
        x = self.relu(x)  # ReLU activation
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu(x)  # ReLU activation
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.relu(x)  # ReLU activation
        x = self.dropout3(x)
        x = self.global_max_pooling(x)
        return x.squeeze(2)
