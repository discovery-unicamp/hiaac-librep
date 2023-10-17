import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from librep.base.transform import Transform
import copy
from librep.estimators.simclr.torch.models.simclr_head import SimCLRHead
from librep.estimators.simclr.torch.dataset_simclr import DatasetSIMCLR



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
        def __init__(self, input_shape,n_components=98, batch_size=256, transform_funcs=[], temperature=1.0, epochs=200, is_transform_function_vectorized=True, verbose=1, patience=10, min_delta=0.001,device="cuda"):
            self.model=simclr_head = SimCLRHead(input_shape,n_components=n_components).to(device)           
            self.device = device
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)  # Initial learning rate is 0.1
            self.temperature = temperature
            self.batch_size = batch_size
            self.transform_funcs = transform_funcs
            self.epochs = epochs
            self.is_transform_function_vectorized = is_transform_function_vectorized
            self.verbose = verbose
            self.epoch_wise_loss = []
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float('inf')
            self.num_epochs_without_improvement = 0

        def fit(self, dataset):
            nt_xent_loss = NTXentLoss(temperature=self.temperature, normalize=True, weights=1.0)

            for epoch in range(self.epochs):
                step_wise_loss = []

                ds = DatasetSIMCLR(dataset, self.transform_funcs, self.device)
                batched_dataset = ds.get_transformed_items(self.batch_size, self.is_transform_function_vectorized)

                for transforms in batched_dataset:
                    transform_1 = transforms[0]
                    transform_2 = transforms[1]
                    #print(transform_1.shape,transform_2.shape)

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
                    print(f'Early stopping after {epoch + 1} epochs with no improvement.')
                    break

            return self.model, self.epoch_wise_loss


        def predict(self, input_data):
            """
            Perform inference (prediction) on input_data using the trained SimCLR model.

            Args:
                input_data (np.ndarray or torch.Tensor): Input data for which predictions should be made.

            Returns:
                np.ndarray: Predicted embeddings for the input_data.
            """
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
               
                embeddings = self.model(input_data)
                embeddings = embeddings.cpu().numpy()

            return embeddings

        def transform(self, X):
            intermediate_model = copy.deepcopy(self.model.base_model)
            test_data = torch.tensor(X, dtype=torch.float32).to(self.device)
            embeddings = intermediate_model(test_data).cpu().detach().numpy()
            return embeddings


