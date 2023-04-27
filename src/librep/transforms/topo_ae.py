import numpy as np
import torch
from librep.estimators.ae.torch.models.topological_ae.topological_ae import (
    TopologicallyRegularizedAutoencoder
)
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike
from torch.optim import Adam
import matplotlib.pyplot as plt
import pickle
import random
import os
import shutil
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
# import torch.nn.DataParallel
# torch.nn.DataParallel(model, device_ids=[0, 1, 2])


class TopologicalDimensionalityReduction(Transform):

    def __init__(
        self, ae_model='ConvolutionalAutoencoder', ae_kwargs=None,
        lam=1., patience=None, num_epochs=500, batch_size=64,
        input_shape=(-1, 1, 28, 28), cuda_device_name='cuda:0',
        start_dim=180, latent_dim=10,
        save_dir='data/', save_tag=0, save_frequency=250, verbose=False
    ):
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        # dist.init_process_group("gloo", rank=0, world_size=6)
        self.save_dir = save_dir
        self.save_tag = save_tag
        self.save_frequency = save_frequency
        self.patience = patience
        self.num_epochs = num_epochs
        self.model_name = ae_model
        self.model_lambda = lam
        self.model_start_dim = start_dim
        self.model_latent_dim = latent_dim
        self.ae_kwargs = ae_kwargs
        self.verbose = verbose
        # Setting cuda device
        self.cuda_device = torch.device(cuda_device_name)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.max_loss = None
        self.current = {
            'epoch': 0,
            'train_recon_error': None,
            'train_topo_error': None,
            'train_error': None,
            'val_recon_error': None,
            'val_topo_error': None,
            'val_error': None,
            'last_error': None
        }
        self.history = {
            'epoch': [],
            'train_recon_error': [],
            'train_topo_error': [],
            'train_error': [],
            'val_recon_error': [],
            'val_topo_error': [],
            'val_error': []
        }

    def fit(self, X: ArrayLike, y: ArrayLike = None, X_val: ArrayLike = None, y_val: ArrayLike = None):
        # Computing input dimensions for the model
        # in the second dim of X.shape
        # ----------------------------------------------
        # When the input is 2d:
        # ----------------------------------------------
        original_dim = X.shape[1]
        # Setting self.input_shape
        self.input_shape = (-1, 1, original_dim)
        if self.ae_kwargs['num_CL'] == 0:
            self.input_shape = (-1, original_dim)
        # Setting ae_kwargs['input_dims']
        self.ae_kwargs['input_dims'] = (1, original_dim)
        # ----------------------------------------------
        # When the input is 3d (length, dim1, dim2): TODO
        # ----------------------------------------------
        
        # Initializing all
        self.model = TopologicallyRegularizedAutoencoder(
            autoencoder_model=self.model_name,
            lam=self.model_lambda, ae_kwargs=self.ae_kwargs
        )
        self.model.to(self.cuda_device)
        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # Save file name
        random_number = random.randint(1000, 9999)
        best_file_name = '{}_{}_{}_{}'.format(
            self.model_name, random_number, self.model_lambda, self.save_tag)
        best_file_name = best_file_name + '.toerase' # custom extension
        
        # First assignation
        train_X = X
        train_Y = y
        val_X = X_val
        val_Y = y_val
        
        # If it is None, then update
        if val_X is None:
            # Splitting X into train and validation
            train_X, val_X, train_Y, val_Y = train_test_split(
                X, y, random_state=0,
                train_size=.8,
                stratify=y
            )
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_X,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_X,
            batch_size=self.batch_size,
            shuffle=True
        )
        patience = self.patience
        max_loss = self.max_loss
        # Preparing for plot
        self.train_final_error = []
        self.train_recon_error = []
        self.train_topo_error = []

        self.val_final_error = []
        self.val_recon_error = []
        self.val_topo_error = []
        # Setting cuda
        # cuda0 = torch.device('cuda:0')
        for epoch in range(self.num_epochs):
            epoch_number = self.current['epoch'] + 1
            epoch_train_loss = []
            epoch_train_ae_loss = []
            epoch_train_topo_error = []
            epoch_val_loss = []
            epoch_val_ae_loss = []
            epoch_val_topo_error = []
            self.model.train()
            for data in train_data_loader:
                # reshaped_data = np.reshape(data, self.input_shape)
                # in_tensor = torch.tensor(reshaped_data, device=self.cuda_device).float()
                in_tensor = torch.reshape(data, self.input_shape)
                in_tensor = in_tensor.to(self.cuda_device)
                in_tensor = in_tensor.float()
                loss, loss_components = self.model(in_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss.append(loss.item())
                epoch_train_ae_loss.append(loss_components['loss.autoencoder'].item())
                epoch_train_topo_error.append(loss_components['loss.topo_error'].item())
            # Verificar despues self.model()
            for data in val_data_loader:
                # reshaped_data = np.reshape(data, self.input_shape)
                # in_tensor = torch.tensor(reshaped_data, device=self.cuda_device).float()
                in_tensor = torch.reshape(data, self.input_shape)
                in_tensor = in_tensor.to(self.cuda_device)
                in_tensor = in_tensor.float()
                loss, loss_components = self.model(in_tensor)
                epoch_val_loss.append(loss.item())
                epoch_val_ae_loss.append(loss_components['loss.autoencoder'].item())
                epoch_val_topo_error.append(loss_components['loss.topo_error'].item())
            self.current['epoch'] = self.current['epoch'] + 1
            self.current['train_recon_error'] = np.mean(epoch_train_ae_loss)
            self.current['train_topo_error'] = np.mean(epoch_train_topo_error)
            self.current['train_error'] = np.mean(epoch_train_loss)
            self.current['val_recon_error'] = np.mean(epoch_val_ae_loss)
            self.current['val_topo_error'] = np.mean(epoch_val_topo_error)
            self.current['val_error'] = np.mean(epoch_val_loss)
            self.history['epoch'].append(self.current['epoch'])
            self.history['train_recon_error'].append(self.current['train_recon_error'])
            self.history['train_topo_error'].append(self.current['train_topo_error'])
            self.history['train_error'].append(self.current['train_error'])
            self.history['val_recon_error'].append(self.current['val_recon_error'])
            self.history['val_topo_error'].append(self.current['val_topo_error'])
            self.history['val_error'].append(self.current['val_error'])
            loss_per_epoch = self.current['val_error']

            # Check for save the BEST version every "n" epochs: save frequency
            # assuming there is already a best version called "best_file_name"
            if epoch_number % self.save_frequency == 0:
                self.partial_save(reuse_file=best_file_name)
                # Copy the file and rename it:
                # shutil.copyfile(self.save_dir + best_file_name, self.save_dir + '')
            
            # Update max loss allowed: if None, then copy from loss_per_epoch
            # print(max_loss)
            max_loss = max_loss or loss_per_epoch
            # If this model beats the better found until now:
            if loss_per_epoch < max_loss:
                # print('MAXLOSS update from', max_loss, 'to', loss_per_epoch, random_number)
                # If LAST model was already created, delete it
                if os.path.exists(best_file_name):
                    os.remove(best_file_name)
                # Save the new LAST model
                self.partial_save(name=best_file_name)
                # Update max_loss
                max_loss = loss_per_epoch
                if self.verbose:
                    print('Best result found at', self.current['epoch'])
            loss_per_epoch = np.mean(epoch_val_ae_loss) + np.mean(epoch_val_topo_error)
            ae_loss_per_epoch = np.mean(epoch_val_ae_loss)
            topo_loss_per_epoch = np.mean(epoch_val_topo_error)
            if self.verbose:
                print(f'Epoch:{epoch+1}, P:{patience}, Loss:{loss_per_epoch:.4f}, Loss-ae:{ae_loss_per_epoch:.4f}, Loss-topo:{topo_loss_per_epoch:.4f}')
            if self.patience:
                if max_loss < loss_per_epoch:
                    if patience == 0:
                        break
                    patience -= 1
                else:
                    max_loss = loss_per_epoch
                    patience = self.patience
        # Update to the best version found
        self.partial_load(name=best_file_name)
        # Erase the temporal file
        os.remove(self.save_dir + best_file_name)
        return self
    
    def plot_training(self, title_plot=None):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_title('Training')
        if title_plot:
            ax.set_title(title_plot)
        ax.plot(self.history['train_recon_error'], label='reconstruction error - train', color='red')
        ax.plot(self.history['val_recon_error'], label='reconstruction error - val', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel("Reconstruction error", color="red", fontsize=14)
        ax.legend(loc=2)
        ax.set_ylim(bottom=0)

        ax2 = ax.twinx()
        ax2.plot(self.history['train_topo_error'], label='Topological error - train', color='blue')
        ax2.plot(self.history['val_topo_error'], label='Topological error - val', color='black')
        ax2.set_ylabel("Topological error", color="blue", fontsize=14)
        ax2.legend(loc=1)
        ax2.set_ylim(bottom=0)
        plt.grid()
        plt.show()
    
    def save(self, save_dir='data/', tag=None):
        model_name = self.model_name
        model_lambda = self.model_lambda
        model_start_dim = self.model_start_dim
        model_latent_dim = self.model_latent_dim
        model_epc = self.current['epoch']
        filename = '{}_{}_{}-{}_{}_{}.pkl'.format(
            model_name, model_lambda,
            model_start_dim, model_latent_dim,
            model_epc, self.save_tag)
        full_dir = self.save_dir + filename
        filehandler = open(full_dir, 'wb')
        pickle.dump(self, filehandler)
        filehandler.close()
        return full_dir
    
    def partial_save(self, name=None, reuse_file=None):
        model_name = self.model_name
        model_lambda = self.model_lambda
        model_start_dim = self.model_start_dim
        model_latent_dim = self.model_latent_dim
        model_epc = self.num_epochs
        model_tag = self.save_tag
        filename = '{}_{}_{}-{}_{}_{}_ep{}'.format(
            model_name, model_lambda,
            model_start_dim, model_latent_dim,
            model_epc, model_tag, self.current['epoch'])
        if name:
            filename = name
        full_dir = self.save_dir + filename
        if reuse_file:
            shutil.copyfile(self.save_dir + reuse_file, full_dir)
            return
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, full_dir)
    
    def partial_load(self, epoch=250, name=None):
        model_name = self.model_name
        model_lambda = self.model_lambda
        model_start_dim = self.model_start_dim
        model_latent_dim = self.model_latent_dim
        model_epc = self.num_epochs
        model_tag = self.save_tag
        filename = '{}_{}_{}-{}_{}_{}_ep{}'.format(
            model_name, model_lambda,
            model_start_dim, model_latent_dim,
            model_epc, model_tag, epoch)
        if name:
            filename = name
        full_dir = self.save_dir + filename
        checkpoint = torch.load(full_dir)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load(self, filename='data/test.pkl'):
        filehandler = open(filename, 'rb')
        self = pickle.load(filehandler)
        filehandler.close()
        print('Loaded ', filename)
    
    # TODO
    def transform(self, X: ArrayLike):
        # Setting cuda
        cuda0 = torch.device('cuda:0')
        self.model.eval()
        reshaped_data = np.reshape(X, self.input_shape)
        in_tensor = torch.tensor(reshaped_data, device=cuda0).float()
        return self.model.encode(in_tensor).cpu().detach().numpy()
    
    def inverse_transform(self, X: ArrayLike):
        # Setting cuda
        cuda0 = torch.device('cuda:0')
        self.model.eval()
        reshaped_data = np.reshape(X, (-1, 1, X.shape[-1]))
        in_tensor = torch.tensor(reshaped_data, device=cuda0).float()
        decoded = self.model.decode(in_tensor).cpu().detach().numpy()
        return np.reshape(decoded, (X.shape[0], -1))

    def transform_and_back(self, X: ArrayLike, plot_function):
        self.model.eval()
        reshaped_data = np.reshape(X, self.input_shape)
        in_tensor = torch.Tensor(reshaped_data).float()
        X_encoded = self.model.encode(in_tensor).detach().numpy()
        plot_function(X, X_encoded)
        return 
    
    def analize_patience(self, data):
        patiences = []
        patience = 0
        p = patience
        max_loss = np.max(data) + 1
        for index in range(1, len(data)):
            # print(index, data[index], p)
            if data[index] < max_loss:
                max_loss = data[index]
                p = patience
            else:
                if p == 0:
                    # print('PATIENCE', patience,' found in index', index, 'with value', data[index])
                    patiences.append(index)
                    patience +=1
                    p += 1
                p -= 1
        return (data, patiences)


class ConvTAETransform(TopologicalDimensionalityReduction):
    def __init__(self,
                 model_name='ConvTAE_def',
                 model_lambda=1,
                 patience=None,
                 num_epochs=175,
                 start_dim=180,
                 latent_dim=2,
                 batch_size=64,
                 cuda_device_name='cuda:0',
                 extra_properties={},
                 save_dir='data/', save_tag=0, save_frequency=250):
        ae_kwargs = {
            'input_dims': (1, start_dim),
            'latent_dim': latent_dim
        }
        ae_kwargs.update(extra_properties)
        input_shape = (-1, 1, start_dim)
        if ae_kwargs['num_CL'] == 0:
            input_shape = (-1, start_dim)
        super().__init__(
            ae_model=model_name,
            ae_kwargs=ae_kwargs,
            lam=model_lambda,
            patience=patience,
            num_epochs=num_epochs,
            batch_size=batch_size,
            input_shape=input_shape,
            cuda_device_name=cuda_device_name,
            start_dim=start_dim,
            latent_dim=latent_dim,
            save_dir=save_dir,
            save_tag=save_tag,
            save_frequency=save_frequency
        )
