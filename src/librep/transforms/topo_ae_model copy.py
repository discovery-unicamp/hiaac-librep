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
        # print('VIEW-FORWARD', x.size(), self.shape)
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


    # def build_encoder_conv_layers(
    #         self, num_conv_layers=0, # basic
    #         kernel=3, stride=1, padding=0, # conv related
    #         pooling_type='none', pooling_kernel=2, pooling_stride=2, # pooling
    #         groups=1, input_size=(6,60), dropout=0.2): # extra
    #     layers = []
    #     out_channels_dict = {
    #         1: [128],
    #         2: [64, 128],
    #         3: [32, 64, 128],
    #         4: [16, 32, 64, 128]
    #     }
    #     def l_out(l_in, padding, kernel_size, stride, dilation=1):
    #         return int(((l_in + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1)
        
    #     for out_channels in out_channels_dict.get(num_conv_layers, []):
    #         # If not enough features, skip
    #         output_input_size = (out_channels, l_out(input_size[1], padding, kernel, stride))    
    #         if output_input_size[1] < 0: break
    #         # Add a new layer to the encoder
    #         conv_layer_to_append = nn.Conv1d(
    #             in_channels=input_size[0],
    #             out_channels=out_channels,
    #             kernel_size=kernel,
    #             stride=stride,
    #             padding=padding,
    #             groups=groups
    #         )
    #         layers.append(conv_layer_to_append)
    #         input_size = output_input_size
    #         # Add a pooling layer to the encoder
    #         if pooling_type != 'none':
    #             poolings = {
    #                 'max': nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_stride),
    #                 'avg': nn.AvgPool1d(kernel_size=pooling_kernel, stride=pooling_stride)
    #             }
    #             pooling_to_append = poolings[pooling_type]
    #             layers.append(pooling_to_append)
    #             # TODO: Update input_size
    #             # input_size = (out_channels, l_out(input_size[1], 0, pooling_kernel, pooling_stride))
    #         # Adding ReLU
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(dropout))
    #     return layers, input_size
    
    # def build_encoder_fc_layers(self, input_size, encoding_size, num_fc_layers=0, dropout=0.2):
    #     layers = []
    #     dimensions = np.linspace(input_size[1]*input_size[0], encoding_size, num_fc_layers+2).round().astype(int)
    #     for index, dim in enumerate(dimensions[:-1]):
    #         layer = nn.Linear(dim, dimensions[index+1])
    #         layers.append(layer)
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(dropout))
    #     # Delete Dropout
    #     layers.pop()
    #     # Delete ReLU
    #     layers.pop()
    #     return layers, dimensions
    
    # def build_decoder_fc_layers(self, output_size, encoding_size, num_fc_layers=0, dropout=0.2):
    #     layers = []
    #     dimensions = np.linspace(encoding_size, output_size[1]*output_size[0], num_fc_layers+2).round().astype(int)
    #     for index, dim in enumerate(dimensions[:-1]):
    #         layer = nn.Linear(dim, dimensions[index+1])
    #         layers.append(layer)
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(dropout))
    #     # Delete Dropout
    #     layers.pop()
    #     # Delete ReLU
    #     layers.pop()
    #     return layers, dimensions
        
    # def build_decoder_conv_layers(
    #         self, num_conv_layers=0, # basic
    #         kernel=3, stride=1, padding=0, # conv related
    #         pooling_type='none', pooling_kernel=2, pooling_stride=2, # pooling
    #         groups=1, input_size=(6,60), dropout=0.2): # extra
    #     layers = []
    #     out_channels_dict = {
    #         1: [128],
    #         2: [128, 64],
    #         3: [128, 64, 32],
    #         4: [128, 64, 32, 16]
    #     }

    #     connection_to_linear = current_input_size[0]*current_input_size[1]
    #     # print('\nSIZE BEFORE LINEAR', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear)
    #     # if num_conv_layers != 0:
    #     # Adding the "View" view
    #     view_layer = View((-1, connection_to_linear))
    #     test_data = view_layer(test_data)
    #     # print('SIZE BEFORE LINEAR', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear)
    #     encoder_layers.append(view_layer)
    #     # Adding the linear layers
    #     dimensions = np.linspace(connection_to_linear, ae_encoding_size, num_fc_layers+2).round().astype(int)
    #     for index, dim in enumerate(dimensions[:-1]):
    #         layer = nn.Linear(dim, dimensions[index+1])
    #         test_data = layer(test_data)
    #         # sizes_to_compare.append(test_data.size())
    #         encoder_layers.append(layer)
    #         encoder_layers.append(nn.ReLU())
    #         encoder_layers.append(nn.Dropout(dropout))
    #     # Delete the last ReLU()
    #     encoder_layers.pop()
    #     encoder_layers.pop()
    #     # print('LATENT', test_data.size(), 'DIMENSIONS', dimensions, '\n')
    #     # Building the encoder
    #     self.encoder = nn.Sequential(*encoder_layers)
            
            # temporal_decoder_layers.append(nn.ReLU())
            # # Testing the conv layer
            # test_data = conv_layer_to_append(test_data)
            # output_padding = current_input_size[-1] + 2*conv_padding - conv_kernel - (test_data.size()[-1]-1)*conv_stride
            # deconv_layer_to_append = nn.ConvTranspose1d(
            #     in_channels=i,
            #     out_channels=in_channels,
            #     kernel_size=conv_kernel,
            #     stride=conv_stride,
            #     padding=conv_padding,
            #     groups=conv_groups,
            #     output_padding=output_padding
            # )
            # temporal_decoder_layers.append(deconv_layer_to_append)
            # # sizes_to_compare.append(test_data.size())
            # # Update values
            # current_input_size = test_data.size()
            # in_channels = test_data.size(0)
            # # If not enough features, skip
            # if current_input_size[1] < pooling_kernel: continue
            # # Add a pooling layer to the encoder
            # if pooling_after != 'none':    
            #     poolings = {
            #         'max': nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_stride),
            #         # 'avg': nn.AvgPool1d(kernel_size=pooling_kernel, stride=pooling_stride)
            #     }
            #     unpoolings = {
            #         'max': nn.MaxUnpool1d(kernel_size=pooling_kernel, stride=pooling_stride),
            #         # 'avg': nn.AvgUnpool1d(kernel_size=pooling_kernel, stride=pooling_stride)
            #     }
            #     pooling_to_append = poolings[pooling_after]
            #     unpooling_to_append = unpoolings[pooling_after]
            #     encoder_layers.append(pooling_to_append)
            #     temporal_decoder_layers.append(unpooling_to_append)
            #     # Testing the pooling
            #     test_data = pooling_to_append(test_data)
            #     # sizes_to_compare.append(test_data.size())
            #     # Update values
            #     current_input_size = test_data.size()
            #     in_channels = test_data.size(0)
            # # Adding ReLU
            # encoder_layers.append(nn.ReLU())
            # encoder_layers.append(nn.Dropout(dropout))
    

    def create_model(self, info: dict):
        # Get the values from the dictionary
        dropout = info.get('ae_dropout', 0)
        
        ae_encoding_size = info.get('ae_encoding_size', 360)
        in_channels = info.get('input_channels', 6)
        current_input_size = info.get('input_size', (6, 60))
        ## Conv related
        num_conv_layers = info.get('ae_conv_num', 0)
        conv_kernel = info.get('ae_conv_kernel', 3)
        conv_stride = info.get('ae_conv_stride', 1)
        conv_padding = info.get('ae_conv_padding', 0)
        pooling_type = info.get('ae_conv_pooling_type', 'none')
        pooling_kernel = info.get('ae_conv_pooling_kernel', 2)
        pooling_stride = info.get('ae_conv_pooling_stride', 2)
        conv_groups = info.get('ae_conv_groups', 1)
        # FC related
        num_fc_layers = info.get('ae_fc_num', 0)
        # DECODER RELATED

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
        test_data = torch.randn(current_input_size)
        for i in conv_channels_sequences[num_conv_layers]:
            # print('ITERATION', i, 'CURRENT SIZE', current_input_size)
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
            
            temporal_decoder_layers.append(nn.ReLU())
            # Testing the conv layer
            test_data = conv_layer_to_append(test_data)
            output_padding = current_input_size[-1] + 2*conv_padding - conv_kernel - (test_data.size()[-1]-1)*conv_stride
            deconv_layer_to_append = nn.ConvTranspose1d(
                in_channels=i,
                out_channels=in_channels,
                kernel_size=conv_kernel,
                stride=conv_stride,
                padding=conv_padding,
                groups=conv_groups,
                output_padding=output_padding
            )
            temporal_decoder_layers.append(deconv_layer_to_append)
            # sizes_to_compare.append(test_data.size())
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
                # sizes_to_compare.append(test_data.size())
                # Update values
                current_input_size = test_data.size()
                in_channels = test_data.size(0)
            # Adding ReLU
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
        # print('AFTER CONV', test_data.size(), 'CURRENT INPUT SIZE', current_input_size)
        connection_to_linear = current_input_size[0]*current_input_size[1]
        # print('\nSIZE BEFORE LINEAR', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear)
        # if num_conv_layers != 0:
        # Adding the "View" view
        view_layer = View((-1, connection_to_linear))
        test_data = view_layer(test_data)
        # print('SIZE BEFORE LINEAR', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear)
        encoder_layers.append(view_layer)
        # Adding the linear layers
        dimensions = np.linspace(connection_to_linear, ae_encoding_size, num_fc_layers+2).round().astype(int)
        for index, dim in enumerate(dimensions[:-1]):
            layer = nn.Linear(dim, dimensions[index+1])
            test_data = layer(test_data)
            # sizes_to_compare.append(test_data.size())
            encoder_layers.append(layer)
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
        # Delete the last ReLU()
        encoder_layers.pop()
        encoder_layers.pop()
        # print('LATENT', test_data.size(), 'DIMENSIONS', dimensions, '\n')
        # Building the encoder
        self.encoder = nn.Sequential(*encoder_layers)
        # print(sizes_to_compare, '\n')
        # --------------------------------------------------------------
        # -------------------- BUILDING THE DECODER --------------------
        # --------------------------------------------------------------
        decoder_layers = []
        # Reversing dimensions
        dimensions = dimensions[::-1]
        for index, dim in enumerate(dimensions[:-1]):
            layer = nn.Linear(dim, dimensions[index+1])
            test_data = layer(test_data)
            # print('AFTER LINEAR', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear)
            decoder_layers.append(layer)
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
        # print('DECODER LAYERS', decoder_layers, '\n')
        # Delete the last Dropout()
        decoder_layers.pop()
        # print('AFTER LINEAR', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear)
        # Adding the "View" view before the last ReLU()
        if num_conv_layers != 0:
            # decoder_layers.pop()
            last_relu = decoder_layers.pop()
            view_layer = View((-1, current_input_size[0], current_input_size[1]))
            test_data = view_layer(test_data)
            # print('AFTER VIEW', test_data.size(), 'CONNECTION TO LINEAR', connection_to_linear, '\n')
            decoder_layers.append(view_layer)
            decoder_layers.append(last_relu)
        # Adding the deconv layers from the temporal array
        # print('\nTESTING CONVTRANS1D - START')
        for layer in temporal_decoder_layers[::-1]:
            test_data = layer(test_data)
            # print('AFTER A CONVTRANS1D', test_data.size(), layer)
            decoder_layers.append(layer)
        # print('TESTING CONVTRANS1D - END')
        # Building the decoder
        self.decoder = nn.Sequential(*decoder_layers)
        # print(self.encoder, '\n')
        # print(self.decoder)
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
        # print('FORWARD', x.size())
        latent = self.encoder(x)
        # print('LATENT', latent.size())
        x_reconst = self.decoder(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        final_error = reconst_error
        if self.ae_topo_lambda > 0:
            topo_error = self.compute_topology(x, latent)
            final_error = reconst_error + self.ae_topo_lambda * topo_error
        else:
            topo_error = 0
        return final_error, reconst_error, topo_error