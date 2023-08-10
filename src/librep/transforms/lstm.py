import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from copy import deepcopy

#########################################################################
# Creating LSTM model
#########################################################################

# the reducer
class LSTM(nn.Module):
  def __init__(self, latent_dim, device, n_layers, input_size):
    super(LSTM, self).__init__()
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
    h_0 = torch.zeros(self.n_layers, x.shape[0], self.latent_dim).float().to(self.device)
    c_0 = torch.zeros(self.n_layers, x.shape[0], self.latent_dim).float().to(self.device)
    h_n, _ = self.lstm(x, (h_0, c_0))
    return h_n[:,-1] # returning the last hidden state

# a model to train the reducer
class TrainingModule(nn.Module):
  def __init__(self, lstm_reducer, n_classes=7):
    super(TrainingModule, self).__init__()
    self.lstm = lstm_reducer
    self.fc = nn.Linear(self.lstm.latent_dim, n_classes)

  def forward(self, x):
    x = self.lstm(x)
    x = self.fc(x)
    x = F.log_softmax(x, dim=1)
    return x

#########################################################################
# Trainer pre-trains the LSTM and saves the weights
#########################################################################
import matplotlib.pyplot as plt
from librep.base.transform import Transform 
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

def save_plot(loss, acc, dataset_name):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
  ax1.plot(loss['train'], label='train')
  ax1.plot(loss['val'], label='validation')
  ax1.set_title('Loss')
  ax1.legend()
  ax2.plot(acc['train'], label='train')
  ax2.plot(acc['val'], label='validation')
  ax2.set_title('Accuracy')
  ax2.legend()
  plt.savefig('./plots/lstm_' + dataset_name + '.png')
  plt.close(fig)
 
class LSTMTrainer(Transform):
  def __init__(self, weight_file=None, n_epochs=40, learning_rate=1e-4,
               batch_size=32, hidden_size=20, n_classes=7, n_layers=1,
               sequence_length=60, input_size=6, patience=5):
    assert weight_file is not None, 'weight_file must be specified'
    
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.input_size = input_size
    self.sequence_length = sequence_length
    self.model = TrainingModule(LSTM(latent_dim=hidden_size, device=self.device, n_layers=n_layers, input_size=input_size), n_classes=n_classes)
    self.epochs = n_epochs
    self.lr = learning_rate
    self.batch_size = batch_size
    self.fit_dataset = weight_file
    self.patience = patience

    print('Using device:', self.device)
    
  # Reshape the dataset into LSTM input 
  def _reshapeToLSTM(self, tensor):
    tensor = torch.reshape(tensor, (tensor.shape[0], self.input_size, self.sequence_length)) # (n_samples, 6, 60)
    return torch.permute(tensor, (0, 2, 1)) # (n_samples, seq_length, H_in)
  
  def __one_epoch(self, dataloader: DataLoader, mode='train'):
    cumulative_loss = 0
    cumulative_corrects = 0
    loss_counter = 0
    for inputs, labels in dataloader:
      inputs = inputs.float().to(self.device)
      labels = labels.type(torch.LongTensor).to(self.device)
      # zero the parameter gradients
      self.optimizer.zero_grad()
      
      if mode == 'train':
        self.model.train()
        outputs = self.model(inputs.float())
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
      else:
        self.model.eval()
        outputs = self.model(inputs.float())
        loss = self.criterion(outputs, labels)
      
      # calculating loss
      cumulative_loss += loss.item() * inputs.size(0)
      loss_counter += 1
      
      # calculating acc
      _, preds = torch.max(outputs, 1)  
      cumulative_corrects += torch.sum(preds == labels.data)
    
    return cumulative_loss / loss_counter, cumulative_corrects.cpu().double() / len(dataloader.dataset)
  
  def _training_loop(self, train_dataloader, val_dataloader):  
    ############### Training setup ##############
    # loss function
    self.criterion = nn.NLLLoss()

    # Observe that all parameters are being optimized
    self.optimizer = Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # sending the model to the GPU
    model = self.model.float().to(self.device)

    ############### Start training ##############
    epoch_loss_train = []
    epoch_loss_val = []
    epoch_acc_train = []
    epoch_acc_val = []
    loss_threshold = np.inf
    patience_counter = 0

    for epoch in range(self.epochs):
      patience_counter += 1
      train_loss, train_acc = self.__one_epoch(train_dataloader, mode='train')
      print(f'EPOCH:{epoch} TRAIN loss: {train_loss:.4f}')
      # if 'validation' in dataloaders:
      validation_loss, val_acc = self.__one_epoch(val_dataloader, mode='validation')
      print(f'EPOCH:{epoch} VALIDATION loss: {validation_loss:.4f}')
      
      epoch_acc_train.append(train_acc)
      epoch_acc_val.append(val_acc)
      epoch_loss_val.append(validation_loss)
      epoch_loss_train.append(train_loss)
      
      # calculating accuracies
      
      
      print('-' * 10)
      if validation_loss < loss_threshold:
        patience_counter = 0
        self.model_best_state_dict = deepcopy(self.model.state_dict())
        loss_threshold = validation_loss
      if self.patience and patience_counter > self.patience:
        break

    self.losses = {
      'train': epoch_loss_train,
      'val': epoch_loss_val
    }
    self.accs = {
      'train': epoch_acc_train,
      'val': epoch_acc_val
    }
    self.model.load_state_dict(self.model_best_state_dict)
    return model
  
  def fit(self, X, y, X_val = None, y_val = None):
    ### If there is no X_val, create it from X
    if X_val is None:
      X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    ### Organizing data    
    x_train = self._reshapeToLSTM(torch.Tensor(X))
    y_train = torch.Tensor(np.array(y))
    x_val = self._reshapeToLSTM(torch.Tensor(X_val))
    y_val = torch.Tensor(np.array(y_val))
    
    train_dataloader = DataLoader(
      TensorDataset(x_train, y_train),
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=0
    )
    val_dataloader = DataLoader(
      TensorDataset(x_val, y_val),
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=0
    )
    
    ############### Start training loop ##############
    self.model = self._training_loop(train_dataloader, val_dataloader)
    self.reducer = self.model.lstm
    
    # saving plots
    save_plot(self.losses, self.accs, self.fit_dataset)
      
    # saving weights
    torch.save(self.reducer.state_dict(), './lstm_weights/lstm_weights_' + self.fit_dataset + '.pt')
    print(f'weights saved to ./lstm_weights/lstm_weights_{self.fit_dataset}.pt')
    
  def transform(self, X):
    X = self._reshapeToLSTM(torch.Tensor(X))
    X = X.float().to(self.device)
    reducer = self.reducer.float().to(self.device)
    h_n = reducer(X)
    return h_n.cpu().detach().numpy()