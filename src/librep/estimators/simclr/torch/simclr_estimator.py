import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from librep.estimators.simclr.torch.models.linear_model import LinearModel
from librep.estimators.simclr.torch.models.simclr_head import SimCLRHead
from librep.estimators.simclr.torch.simclr import SimCLR
import librep.estimators.simclr.torch.simclr_utils as s_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
from librep.base.estimator import Estimator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Simclr_Estimator(Estimator):
    def __init__(self,trained_simclr_model,input_shape,
                batch_size_head,transform_funcs,temperature_head,epochs_head,
                 save_model=False,verbose=0,total_epochs=50,
                 batch_size=32,lr=0.001,classificator='full',sequence_length=60, input_size=6):
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.loaded_simclr_model=trained_simclr_model
        self.batch_size=batch_size
        self.lr=lr
        self.total_epochs=total_epochs
        self.verbose=verbose
        self.classificador=classificator
        self.input_shape=(sequence_length,input_size)
        self.sequence_length=sequence_length
        self.input_size=input_size
        
        
        if trained_simclr_model==None:
            simclr_head = SimCLRHead(self.input_shape).to(device)
            self.simclr = SimCLR(model=simclr_head,batch_size=batch_size_head,
                                        transform_funcs=transform_funcs,
                                        temperature=temperature_head, epochs=epochs_head,
                                        is_transform_function_vectorized=True,
                                        verbose=verbose,device=device)
        else:
            self.simclr=trained_simclr_model
            
            

        
    def fit(self,X,y, X_val = None, y_val = None):
        print("XXX",X.shape,y.shape)

        ### SimCLR HEAD
        X,X_val,y,y_val=s_utils.get_resize_data(X,X_val,y,y_val,self.input_shape)
        trained_simclr_model,epoch_wise_loss = self.simclr.fit(X)
        print(X_val.shape,y_val.shape)
        y_train = torch.Tensor(np.array(y))
        y_val = torch.Tensor(np.array(y_val))
        num_classes = len(torch.unique(y_train))
                        
        train_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), y_train)
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), y_val)
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        if self.classificador=='full':            
            evaluation_model = LinearModel(trained_simclr_model, num_classes, ).to(self.device)
        self.model=evaluation_model
        

        # Define loss and optimizer
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

            # Save the model if it's the best model so far
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.model=evaluation_model
               # torch.save(evaluation_model.state_dict(), best_model_path)
               # print(f"Best model saved with val accuracy: {best_accuracy:.4f}")

        print("Training finished.")
        # Load the saved model state dict
        #self.model.load_state_dict(torch.load(best_model_path))
        return self
       
    def predict(self, X):
        X = torch.Tensor(X).reshape(-1, self.sequence_length, self.input_size).float().to(self.device)
        outputs = self.model(X)
        _, preds = torch.max(outputs, 1)
        return preds.cpu().detach().numpy()