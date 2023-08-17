import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAELSTM_part_ConvAE(nn.Module):
    def __init__(self, start_dim, n_channels, kernel_size=3, stride=2, padding=0, filter_size=64):
        super().__init__()
        self.start_dim = start_dim
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=n_channels,out_channels=filter_size,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ConvTranspose1d(in_channels=filter_size,out_channels=n_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.ReLU())
        print('MODEL', self.model)
        
    def forward(self, x):
        return self.model(x)
        

class ConvAELSTM_part_LSTM(nn.Module):
    def __init__(self, latent_dim, device, n_layers, input_size):
        super(ConvAELSTM_part_LSTM, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            hidden_size = latent_dim,
            input_size=input_size,
            num_layers=n_layers,
            batch_first=True
            )
        self.latent_dim=latent_dim
        
    def forward(self, x):
        # h_0 = torch.zeros(self.n_layers, x.shape[0], self.latent_dim).float().to(self.device)
        # c_0 = torch.zeros(self.n_layers, x.shape[0], self.latent_dim).float().to(self.device)
        h_n, _ = self.lstm(x)
        return h_n[:,-1] # returning the last hidden state


class ConvAELSTM_part_Softmax(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.fully_connected = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        x = self.fully_connected(x)
        x = F.log_softmax(x, dim=1)
        return x
    

class ConvAELSTM_full(nn.Module):
    def __init__(self, latent_dim, device, n_layers, input_size, n_classes, sequence_length=60):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.conv = ConvAELSTM_part_ConvAE(start_dim=60, n_channels=input_size)
        self.lstm = ConvAELSTM_part_LSTM(latent_dim=latent_dim, device=device, n_layers=n_layers, input_size=input_size)
        self.softmax = ConvAELSTM_part_Softmax(latent_dim=latent_dim, n_classes=n_classes)
    
    # Reshape the dataset into LSTM input 
    def _reshapeToLSTM(self, tensor):
        # tensor = torch.reshape(tensor, (tensor.shape[0], self.input_size, self.sequence_length)) # (n_samples, 6, 60)
        return torch.permute(tensor, (0, 2, 1)) # (n_samples, seq_length, H_in)


    def forward(self, x):
        # print('x before conv', x.shape)
        x = self.conv(x)
        # print('x after conv', x.shape)
        x = self.lstm(self._reshapeToLSTM(x))
        x = self.softmax(x)
        return x
    
    def reduce_dim(self, x):
        self.conv.eval()
        self.lstm.eval()
        print('x before conv', x.shape)
        x = self.conv(x)
        print('x after conv', x.shape)
        x = self.lstm(self._reshapeToLSTM(x))
        return x