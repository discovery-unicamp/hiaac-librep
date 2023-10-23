import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from librep.base.transform import Transform
import copy,os
from librep.estimators.simclr.torch.models.simclr_head import SimCLRHead
from librep.estimators.simclr.torch.dataset_simclr import DatasetSIMCLR
from librep.config.type_definitions import ArrayLike
import librep.estimators.simclr.torch.simclr_utils as s_utils


class NTXentLoss(nn.Module):
    def __init__(self, temperature=1.0, normalize=True, weights=1.0):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.weights = weights

    def forward(self, samples_transform_1, samples_transform_2, model):
        hidden_features_transform_1 = model(samples_transform_1)
        hidden_features_transform_2 = model(samples_transform_2)
        loss = self.calculate_loss(hidden_features_transform_1, hidden_features_transform_2)
        gradients = self.calculate_gradients(loss, model)
        return loss, gradients

    def calculate_loss(self, h1, h2):
        entropy_function = nn.CrossEntropyLoss(reduction='none')

        if self.normalize:
            h1 = F.normalize(h1, dim=1)
            h2 = F.normalize(h2, dim=1)

        batch_size = h1.size(0)
        labels = torch.arange(batch_size, device=h1.device)
        masks = torch.eye(batch_size, device=h1.device)
        LARGE_NUM = torch.tensor(1e9, dtype=torch.float32, device=h1.device)
        masks = torch.eye(batch_size, device=h1.device)
        logits_aa = torch.matmul(h1, h1.t()) / self.temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(h2, h2.t()) / self.temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(h1, h2.t()) / self.temperature
        logits_ba = torch.matmul(h2, h1.t()) / self.temperature
        logits = torch.cat([torch.cat([logits_ab, logits_aa], dim=1), torch.cat([logits_ba, logits_bb], dim=1)], dim=0)
        targets = torch.cat([labels, labels], dim=0)
        loss = entropy_function(logits, targets)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss

    def calculate_gradients(self, loss, model):
        gradients = torch.autograd.grad(loss, model.parameters())
        return gradients

    
class SimCLR(Transform):
        def __init__(self, dataset,input_shape,n_components=98, batch_size=256, transform_funcs=[], temperature=1.0, epochs=200,  verbose=1, patience=10, min_delta=0.001,device='cuda',save_simclr_model=True):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.model=SimCLRHead(input_shape,n_components=n_components).to(self.device)           
            
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)  # Initial learning rate is 0.1
            self.temperature = round(temperature, 2)
            self.batch_size = batch_size
            self.transform_funcs = transform_funcs
            self.epochs = epochs
            self.is_transform_function_vectorized = True
            self.verbose = verbose
            self.epoch_wise_loss = []
            self.patience = patience
            self.min_delta = round(min_delta, 3)
            self.best_loss = float('inf')
            self.num_epochs_without_improvement = 0
            self.input_shape=input_shape
            self.save_simclr_model=save_simclr_model
            self.simclr_model_save_path =dataset+"_"+str(input_shape)+"_"+           str(n_components)+"_"+str(batch_size)+"_"+str(transform_funcs)+"_"+str(self.temperature)+"_"+ str(epochs)+"_"+ str(patience)+"_"+ str(self.min_delta)+"_simclr.pth"
            self.working_directory="../../models/"
            self.simclr_model_save_path = f"{self.working_directory}{self.simclr_model_save_path}"

        def fit(self,X: ArrayLike, y = None, X_val=None, y_val = None):
            X=s_utils.resize_data(X, self.input_shape)
            dataset=X

            if os.path.exists(self.simclr_model_save_path):
                print("exist model")
                self.model.load_state_dict(torch.load(self.simclr_model_save_path))
                self.model.eval() 
                return self.model,[]
            else:                
                nt_xent_loss = NTXentLoss(temperature=self.temperature, normalize=True, weights=1.0)    
                for epoch in range(self.epochs):
                    step_wise_loss = []
                    ds = DatasetSIMCLR(dataset, self.transform_funcs, self.device)
                    batched_dataset = ds.get_transformed_items(self.batch_size, self.is_transform_function_vectorized)
    
                    for transforms in batched_dataset:
                        transform_1 = transforms[0]
                        transform_2 = transforms[1]
                        self.optimizer.zero_grad()
                        loss, gradients = nt_xent_loss(transform_1, transform_2, self.model)
    
                        self.optimizer.zero_grad()
                        for param, grad in zip(self.model.parameters(), gradients):
                            param.grad = grad
                        self.optimizer.step()
    
                        step_wise_loss.append(loss.item())
    
                    epoch_loss = np.mean(step_wise_loss)
                    self.epoch_wise_loss.append(epoch_loss)
    
                    if self.verbose > 0:
                        print("epoch: {} loss: {:.3f}".format(epoch + 1, epoch_loss))
    
                    if epoch_loss < self.best_loss - self.min_delta:
                        self.best_loss = epoch_loss
                        self.num_epochs_without_improvement = 0
                    else:
                        self.num_epochs_without_improvement += 1
    
                    if self.num_epochs_without_improvement >= self.patience:
                        if self.verbose > 0:
                            print(f'Early stopping after {epoch + 1} epochs with no improvement.')
                        break
    
                if (self.save_simclr_model):
                    try:
                        os.makedirs(self.working_directory)
                    except:
                        exist=1
                        
                    
                    torch.save(self.model.state_dict(), self.simclr_model_save_path)
    
                return self.model, self.epoch_wise_loss


        def predict(self, input_data):
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)               
                embeddings = self.model(input_data)
                embeddings = embeddings.cpu().numpy()
            return embeddings

        def transform(self, X):
            X = s_utils.resize_data(X, self.input_shape)
            intermediate_model = copy.deepcopy(self.model.base_model)
            test_data = torch.tensor(X, dtype=torch.float32).to(self.device)
            embeddings = intermediate_model(test_data).cpu().detach().numpy()
            return embeddings


