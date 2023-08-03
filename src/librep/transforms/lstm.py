import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################################################
# Defining constants
#########################################################################
SEQUENCE_LENGTH = 60
INPUT_SIZE = 6
N_CLASSES = 7
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_LAYERS = 1 # stacked LSTMs

#########################################################################
# Creating LSTM model
#########################################################################

# the reducer
class LSTM(nn.Module):
  def __init__(self, hidden_size):
    super(LSTM, self).__init__()
    self.lstm = nn.LSTM(
        hidden_size = hidden_size,
        input_size=INPUT_SIZE,
        num_layers=N_LAYERS,
        batch_first=True
        )
    self.hidden_size=hidden_size

  def forward(self, x):
    h_0 = torch.zeros(N_LAYERS, x.shape[0], self.hidden_size).double().to(DEVICE)
    c_0 = torch.zeros(N_LAYERS, x.shape[0], self.hidden_size).double().to(DEVICE)
    h_n, _ = self.lstm(x, (h_0, c_0))
    return h_n[:,-1] # returning the last hidden state

# a model to train the reducer
class TrainingModule(nn.Module):
  def __init__(self, lstm_reducer):
    super(TrainingModule, self).__init__()
    self.lstm = lstm_reducer
    self.fc = nn.Linear(self.lstm.hidden_size, N_CLASSES)

  def forward(self, x):
    x = self.lstm(x)
    x = self.fc(x)
    x = F.log_softmax(x, dim=1)
    return x

#########################################################################
# Trainer pre-trains the LSTM and saves the weights
#########################################################################
from librep.base.transform import Transform 
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
 
class LSTMTrainer(Transform):
  def __init__(self, weight_file=None, n_epochs=40, learning_rate=1e-4, batch_size=32, hidden_size=20):
    assert weight_file is not None, 'weight_file must be specified'
    
    self.model = TrainingModule(LSTM(hidden_size))
    self.epochs = n_epochs
    self.lr = learning_rate
    self.batch_size = batch_size
    self.fit_dataset = weight_file

    print('Using device:', DEVICE)
    
  # Reshape the dataset into LSTM input 
  def _reshapeToLSTM(self, tensor):
    tensor = torch.reshape(tensor, (tensor.shape[0], INPUT_SIZE, SEQUENCE_LENGTH)) # (n_samples, 6, 60)
    return torch.permute(tensor, (0, 2, 1)) # (n_samples, seq_length, H_in)
  
  def _training_loop(self, dataloaders, dataset_sizes):  
    ############### Training setup ##############
    # loss function
    criterion = nn.NLLLoss()

    # Observe that all parameters are being optimized
    optimizer = Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # sending the model to the GPU
    model = self.model.double().to(DEVICE)

    ############### Start training ##############
    epoch_loss_train = []
    epoch_loss_val = []
    epoch_acc_train = []
    epoch_acc_val = []

    # only run validation if there is a validation set
    if self.validation:
      phases = ['train', 'validation']
    else:
      phases = ['train']  

    for epoch in range(self.epochs):
      for phase in phases:
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
          inputs = inputs.double().to(DEVICE)
          labels = labels.type(torch.LongTensor).to(DEVICE)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward prop
          with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs.double())
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # back prop
            if phase == 'train':
              loss.backward()
              optimizer.step()

          # statistics
          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        print('{} loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if phase == 'train':
          epoch_loss_train.append(epoch_loss)
          epoch_acc_train.append(epoch_acc)
        else:
          epoch_loss_val.append(epoch_loss)
          epoch_acc_val.append(epoch_acc)

      print('End of epoch {}/{}'.format(epoch, self.epochs - 1))
      print('-' * 10)

    self.losses = {
      'train': epoch_loss_train,
      'val': epoch_loss_val
    }
    self.accs = {
      'train': epoch_acc_train,
      'val': epoch_acc_val
    }

    return model
  
  def fit(self, X, y, X_val = None, y_val = None):    
    ############### Organizing data ##############    
    x_train = self._reshapeToLSTM(torch.Tensor(X))    
    y_train = torch.Tensor(y)
    
    inputs, labels = dict(), dict()  
    if X_val is not None and y_val is not None:
      x_val = self._reshapeToLSTM(torch.Tensor(x_val))
      y_val = torch.Tensor(y_val) 
      
      inputs = { 'train': x_train, 'validation': x_val }
      labels = { 'train': y_train, 'validation': y_val }
      phases = ['train', 'validation']
      self.validation = True
      
    else:
      inputs = { 'train': x_train }
      labels = { 'train': y_train }
      phases = ['train']
      self.validation = False
    
    # creating the dataloaders
    datasets, dataloaders, dataset_sizes = dict(), dict(), dict()
    for phase in phases:
      datasets[phase] = TensorDataset(inputs[phase], labels[phase])
      dataloaders[phase] = DataLoader(datasets[phase], batch_size=self.batch_size, shuffle=True, num_workers=0)
      dataset_sizes[phase] = len(datasets[phase])      
    
    ############### Start training loop ##############
    self.model = self._training_loop(dataloaders, dataset_sizes)
    self.reducer = self.model.lstm
    
    # saving weights
    torch.save(self.reducer.state_dict(), './lstm_weights/lstm_weights_' + self.fit_dataset + '.pt')
    print(f'weights saved to ./lstm_weights/lstm_weights_{self.fit_dataset}.pt')
    
  def transform(self, X):
    X = self._reshapeToLSTM(torch.Tensor(X))
    X = X.double().to(DEVICE)
    reducer = self.reducer.double().to(DEVICE)
    h_n = reducer(X)
    return h_n.cpu().detach().numpy()