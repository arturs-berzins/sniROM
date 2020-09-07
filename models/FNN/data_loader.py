"""
Data loader for torch.
"""
# Author: Arturs Berzins <berzins@cats.rwth-aachen.de>
# License: BSD 3 clause

import torch
import torch.utils.data

def create_loader(features_np, targets_np, batch_size, shuffle):
    # https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets

    # Convert from numpy array to torch tensor
    features = torch.from_numpy(features_np).float()
    targets = torch.from_numpy(targets_np).float()
    
    # Create dataset from torch tensors
    dataset = torch.utils.data.TensorDataset(features, targets)
    
    # Create data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=shuffle, num_workers=0)
    return loader
