import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from librep.estimators.simclr.torch.models.model_full import Model_Full
from librep.transforms.simclr import SimCLR
from librep.base.estimator import Estimator
import librep.estimators.simclr.torch.simclr_utils as s_utils

class Simclr_Full__FusionEstimator(Estimator):
    def __init__(self,
                 dataset,
                 input_shape,
                 n_components,   
                 batch_size_head,
                 transform_funcs,
                 temperature_head,
                 epochs_head,
                 patience,
                 min_delta,
                 device,
                 save_reducer,
                 save_model,
                 verbose,                 
                 total_epochs,
                 batch_size,
                 lr,
                ):   
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size=batch_size
        self.lr=lr
        self.total_epochs=total_epochs
        self.verbose=verbose
        self.input_shape=input_shape
        self.sequence_length=self.input_shape[0]
        self.input_size=self.input_shape[1]
        temperature_head=round(temperature_head, 2)
        min_delta=round(temperature_head, 3)
        self.simclr1 = SimCLR(input_shape=input_shape,
                             n_components=n_components,
                             batch_size=batch_size_head, 
                             transform_funcs=transform_funcs[0],
                             temperature=temperature_head,
                             epochs=epochs_head,
                             patience=patience,
                             min_delta=min_delta,
                             save_reducer=save_reducer,
                             
                             verbose=verbose,device=self.device,dataset=dataset)
        
        self.simclr2 = SimCLR(input_shape=input_shape,
                             n_components=n_components,
                             batch_size=batch_size_head, 
                             transform_funcs=transform_funcs[1],
                             temperature=temperature_head,
                             epochs=epochs_head,
                             patience=patience,
                             min_delta=min_delta,
                             save_reducer=save_reducer,
                             
                             verbose=verbose,device=self.device,dataset=dataset)
     
    def fit(self,X,y, X_val = None, y_val = None):
        X,X_val,y,y_val=s_utils.get_resize_data(X,X_val,y,y_val,self.input_shape)
        trained_simclr_model1,epoch_wise_loss1 = self.simclr1.fit(X) 
        trained_simclr_model2,epoch_wise_loss2 = self.simclr2.fit(X) 

        
        y_train = torch.Tensor(np.array(y))
        y_val = torch.Tensor(np.array(y_val))
        num_classes = len(torch.unique(y_train))                        
        train_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), y_train)
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), y_val)
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)       
                  
        evaluation_model = Model_Full(trained_simclr_model1, num_classes, ).to(self.device)
        evaluation_model.load_state_dict(trained_simclr_model2.state_dict(), strict=False)

# Carregue os pesos do segundo modelo
        evaluation_model.load_state_dict(trained_simclr_model1.state_dict(), strict=False)

        
        self.model=evaluation_model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(evaluation_model.parameters(), self.lr)
        best_accuracy = 0.0
        best_model_path = "best_linear_evaluation_model_est.pth"

        for epoch in range(self.total_epochs):
            evaluation_model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.long().to(self.device)
                optimizer.zero_grad()
                outputs = evaluation_model(inputs).float() 
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if(self.verbose):
                print(f"Epoch [{epoch+1}/{self.total_epochs}] - Loss: {running_loss / len(train_loader)}")

            # Evaluate the model on the val set
            evaluation_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.long().to(self.device)
                    outputs = evaluation_model(inputs).float() 
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = correct / total
            if(self.verbose):
                print(f"Val Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.model=evaluation_model
 

        print("Training finished.")
        return self
       
    def predict(self, X):
        X = torch.Tensor(X).reshape(-1, self.sequence_length, self.input_size).float().to(self.device)
        outputs = self.model(X)
        _, preds = torch.max(outputs, 1)
        return preds.cpu().detach().numpy()