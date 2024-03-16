import torch.nn as nn
from librep.estimators.ae.torch.models.topological_ae.topological_signature_distance import TopologicalSignatureDistance
from torch import norm as torch_norm
import torch
import numpy as np


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class ConvTAEModule(nn.Module):

    def __init__(self, info={}):
        super(ConvTAEModule, self).__init__()
        self.info = info
        # info['input_channels']
        # info['input_dim1']
        # info['input_dim2']
        # Autoencoder
        self.create_model(info) # 6,60
        # Topology
        self.ae_topo_lambda = info['ae_topo_lambda'] if 'ae_topo_lambda' in info else 0
        self.topo_sig = TopologicalSignatureDistance()
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1), requires_grad=True)


    def create_model(self, info):
        num_conv_layers = info['ae_conv_num'] if 'ae_conv_num' in info else 0
        # last_conv_channel = info['ae_last_cl_size']
        conv_kernel = info['ae_conv_kernel'] if 'ae_conv_kernel' in info else 3
        ae_encoding_size = info['ae_encoding_size'] # LATENT_DIM
        in_channels = info['input_channels'] if 'input_channels' in info else 6
        current_input_size = info['input_size'] if 'input_size' in info else (6, 60)
        # Conv related
        conv_stride = info['ae_conv_stride'] if 'ae_conv_stride' in info else 1
        conv_padding = info['ae_conv_padding'] if 'ae_conv_padding' in info else 0
        pooling_after = info['ae_conv_pooling_type'] if 'ae_conv_pooling_type' in info else 'none'
        pooling_kernel = info['ae_conv_pooling_kernel'] if 'ae_conv_pooling_kernel' in info else 2
        pooling_stride = info['ae_conv_pooling_stride'] if 'ae_conv_pooling_stride' in info else 2
        conv_groups = info['ae_conv_groups'] if 'ae_conv_groups' in info else 1
        # FC related
        num_fc_layers = info['ae_fc_num'] if 'ae_fc_num' in info else 0
        # num_fc_layers = num_fc_layers if num_conv_layers > 0 else 1
        # Comentado porque pasar de 360 a 0 no cuenta como layer
        # DECODER RELATED
        # decoder_fc_layers = max(0, num_fc_layers-1)
        # decoder_conv_layers = max(0, num_conv_layers-1)


        # Defining possible channel sequences, depending on the number of conv layers
        
        # --------------------------------------------------------------
        # -------------------- BUILDING THE ENCODER --------------------
        # --------------------------------------------------------------
        conv_channels_sequences = {
            0: [],
            1: [256],
            2: [128, 256],
            3: [64, 128, 256],
            4: [32, 64, 128, 256]
        }
        encoder_layers = []
        temporal_decoder_layers = []
        for i in conv_channels_sequences[num_conv_layers]:
            print('ITERATION', i, 'CURRENT SIZE', current_input_size)
            # If not enough features, skip
            if current_input_size[1] < conv_kernel: continue
            # Add a new layer to the encoder
            conv_layer_to_append = nn.Conv1d(
                in_channels=in_channels,
                out_channels=i,
                kernel_size=conv_kernel,
                stride=conv_stride,
                padding=conv_padding,
                groups=conv_groups
            )
            encoder_layers.append(conv_layer_to_append)
            deconv_layer_to_append = nn.ConvTranspose1d(
                in_channels=i,
                out_channels=in_channels,
                kernel_size=conv_kernel,
                stride=conv_stride,
                padding=conv_padding,
                groups=conv_groups,
                output_padding=1 if conv_padding > 3 else 0
            )
            temporal_decoder_layers.append(nn.ReLU())
            temporal_decoder_layers.append(deconv_layer_to_append)
            # Testing the conv layer
            test_data = torch.randn(current_input_size)
            test_data = conv_layer_to_append(test_data)
            # Update values
            current_input_size = test_data.size()
            in_channels = test_data.size(0)
            # If not enough features, skip
            if current_input_size[1] < pooling_kernel: continue
            # Add a pooling layer to the encoder
            if pooling_after != 'none':    
                poolings = {
                    'max': nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_stride),
                    # 'avg': nn.AvgPool1d(kernel_size=pooling_kernel, stride=pooling_stride)
                }
                unpoolings = {
                    'max': nn.MaxUnpool1d(kernel_size=pooling_kernel, stride=pooling_stride),
                    # 'avg': nn.AvgUnpool1d(kernel_size=pooling_kernel, stride=pooling_stride)
                }
                pooling_to_append = poolings[pooling_after]
                unpooling_to_append = unpoolings[pooling_after]
                encoder_layers.append(pooling_to_append)
                temporal_decoder_layers.append(unpooling_to_append)
                # Testing the pooling
                test_data = pooling_to_append(test_data)
                # Update values
                current_input_size = test_data.size()
                in_channels = test_data.size(0)
            # Adding ReLU
            encoder_layers.append(nn.ReLU())
        connection_to_linear = current_input_size[0]*current_input_size[1]
        print('\nSIZE BEFORE LINEAR', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear)
        # Adding the "View" view
        view_layer = View((-1, connection_to_linear))
        test_data = view_layer(test_data)
        print('SIZE BEFORE LINEAR', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear)
        encoder_layers.append(view_layer)
        # Adding the linear layers
        dimensions = np.linspace(connection_to_linear, ae_encoding_size, num_fc_layers+2).round().astype(int)
        for index, dim in enumerate(dimensions[:-1]):
            layer = nn.Linear(dim, dimensions[index+1])
            test_data = layer(test_data)
            encoder_layers.append(layer)
            encoder_layers.append(nn.ReLU())
        # Delete the last ReLU()
        encoder_layers.pop()
        print('LATENT', test_data.size(), 'DIMENSIONS', dimensions, '\n')
        # Building the encoder
        self.encoder = nn.Sequential(*encoder_layers)
        # --------------------------------------------------------------
        # -------------------- BUILDING THE DECODER --------------------
        # --------------------------------------------------------------
        decoder_layers = []
        # Reversing dimensions
        dimensions = dimensions[::-1]
        for index, dim in enumerate(dimensions[:-1]):
            layer = nn.Linear(dim, dimensions[index+1])
            test_data = layer(test_data)
            decoder_layers.append(layer)
            decoder_layers.append(nn.ReLU())
        print('AFTER LINEAR', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear)
        # Adding the "View" view before the last ReLU()
        if num_conv_layers != 0:
            last_relu = decoder_layers.pop()
            view_layer = View((-1, current_input_size[0], current_input_size[1]))
            test_data = view_layer(test_data)
            print('AFTER VIEW', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear, '\n')
            decoder_layers.append(view_layer)
            decoder_layers.append(last_relu)
        # Adding the deconv layers from the temporal array
        print('\nTESTING CONVTRANS1D - START')
        for layer in temporal_decoder_layers[::-1]:
            test_data = layer(test_data)
            print('AFTER A CONVTRANS1D', test_data.size())
            decoder_layers.append(layer)
        print('TESTING CONVTRANS1D - END')
        # Building the decoder
        self.decoder = nn.Sequential(*decoder_layers)
        print(self.encoder, '\n')
        print(self.decoder)
        self.reconst_error = nn.MSELoss()
        ################################################################

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch_norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances
    
    def compute_topology(self, x, latent):
        """Compute the loss of the Topologically regularized autoencoder.
        Args:
            x: Input data
        Returns:
            Tuple of final_loss, (...loss components...)
        """
        x_distances = self._compute_distance_matrix(x)

        dimensions = x.size()
        if len(dimensions) == 4:
            # If we have an image dataset, normalize using theoretical maximum
            batch_size, ch, b, w = dimensions
            # Compute the maximum distance we could get in the data space (this
            # is only valid for images wich are normalized between -1 and 1)
            max_distance = (2**2 * ch * b * w) ** 0.5
            x_distances = x_distances / max_distance
        else:
            # Else just take the max distance we got in the batch
            x_distances = x_distances / x_distances.max()

        latent_distances = self._compute_distance_matrix(latent)
        latent_distances = latent_distances / self.latent_norm

        # Use reconstruction loss of autoencoder
        # ae_loss, ae_loss_comp = self.autoencoder(x)
        topo_error, topo_error_components = self.topo_sig(
            x_distances, latent_distances)

        # normalize topo_error according to batch_size
        batch_size = dimensions[0]
        topo_error = topo_error / float(batch_size) 
        # loss = ae_loss + self.lam * topo_error
        # loss_components = {
        #     'loss.autoencoder': ae_loss,
        #     'loss.topo_error': topo_error
        # }
        # loss_components.update(topo_error_components)
        # loss_components.update(ae_loss_comp)
        return topo_error
    
    def forward(self, x):
        latent = self.encoder(x)
        x_reconst = self.decoder(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        final_error = reconst_error
        if self.ae_topo_lambda > 0:
            topo_error = self.compute_topology(x, latent)
            final_error = reconst_error + self.ae_topo_lambda * topo_error
        else:
            topo_error = 0
        return final_error, reconst_error, topo_error