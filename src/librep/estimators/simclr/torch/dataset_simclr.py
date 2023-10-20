import numpy as np
from librep.estimators.simclr.torch.sensor_data_transformer import SensorDataTransformer
import torch
import matplotlib.pyplot as plt

def random_shuffle_indices(length):
    indices = np.arange(length)
    np.random.shuffle(indices)
    return indices

def ceiling_division(numerator, denominator):
    return -(numerator // -denominator)

def batched_data_generator(data, batch_size):
    num_batches = ceiling_division(data.shape[0], batch_size)
    for i in range(num_batches):
        yield data[i * batch_size : (i + 1) * batch_size]
        
        
class DatasetSIMCLR:
    def __init__(self, dataset, transform_names, device):
        self.data = dataset
        self.device = device
        transformer = SensorDataTransformer()
        self.composite_transform = transformer.get_transform_function(transform_names)

    def get_transformed_items(self, batch_size, is_transform_function_vectorized):
        shuffled_indices = random_shuffle_indices(len(self.data))
        shuffled_dataset = self.data[shuffled_indices]
        batched_dataset = batched_data_generator(shuffled_dataset, batch_size)
                                    
        self.transformed_data_list = []
        for data_batch in batched_dataset:
            if is_transform_function_vectorized:
                transform_1 = self.composite_transform(data_batch)
                transform_2 = self.composite_transform(data_batch)
            else:
                transform_1 = torch.stack([torch.tensor(self.composite_transform(data), dtype=torch.float32) for data in data_batch])
                transform_2 = torch.stack([torch.tensor(self.composite_transform(data), dtype=torch.float32) for data in data_batch])
                
            
            transform_1 = torch.tensor(transform_1, dtype=torch.float32).to(self.device)
            transform_2 = torch.tensor(transform_2, dtype=torch.float32).to(self.device)
            
            
                
            self.transformed_data_list.append((transform_1, transform_2))

        return self.transformed_data_list
    
    
    def get_original_transformed_items(self, batch_size, is_transform_function_vectorized):
        shuffled_indices = random_shuffle_indices(len(self.data))
        shuffled_dataset = self.data[shuffled_indices]
        batched_dataset = batched_data_generator(shuffled_dataset, batch_size)
                                    
        self.transformed_data_list = []
        for data_batch in batched_dataset:
            if is_transform_function_vectorized:
                transform_1 = self.composite_transform(data_batch)
                transform_2 = self.composite_transform(data_batch)
            else:
                transform_1 = torch.stack([torch.tensor(self.composite_transform(data), dtype=torch.float32) for data in data_batch])
                transform_2 = torch.stack([torch.tensor(self.composite_transform(data), dtype=torch.float32) for data in data_batch])
                
            
            transform_1 = torch.tensor(transform_1, dtype=torch.float32).to(self.device)
            transform_2 = torch.tensor(transform_2, dtype=torch.float32).to(self.device)
            
            
                
            self.transformed_data_list.append((data_batch,transform_1, transform_2))

        return self.transformed_data_list

    
    def random_shuffle_indices(self, length):
        indices = np.arange(length)
        np.random.shuffle(indices)
        return indices
