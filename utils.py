"""
Contains utilities.
Responsible for directory structure, saving and loading, snapshot manipulation.
"""
from config import *
from models import RBF, GPR, FNN
models = {
    'RBF':  RBF,
    'GPR':  GPR,
    'FNN':  FNN,
    }
    
import numpy as np
import pandas as pd
import time
import warnings
import pickle
import os
join = os.path.join


def set_up_dataset_directories():
    """
    Set up directory structure for dataset directories
    """
    for dataset in datasets:
        path = join(os.getcwd(),'dataset',dataset)
        os.makedirs(path, exist_ok=True)
    if os.listdir(path):
        print("Directory is empty")

def set_up_data_directories():
    """
    Set up directory structure for data directories
    """
    subdirs = ['features','targets','POD','scaler','errors']
    for subdir in subdirs:
        path = join(os.getcwd(),'data',subdir)
        os.makedirs(path, exist_ok=True)
    for component in components:
        path = join(os.getcwd(),'data','POD',component)
        os.makedirs(path, exist_ok=True)

# model_dir is accessed by the models for storage
model_dir = join(os.getcwd(),'data','models')
def set_up_model_directories():
    """
    Set up directory structure for model directories
    """
    for model_key in models:
        for component in components:
            path = join(model_dir,model_key,component)
            os.makedirs(path, exist_ok=True)
        

def reduce(v, component):
    m = v.shape[0] // len(mask)
    assert m * len(mask) == v.shape[0], "Mask is conflicting with snapshot dimensions."
    tile = (np.asarray(mask)==component)
    ma = np.tile(tile, m)
    return v[ma]

def expand(v, component):
    tile = (np.asarray(mask)==component)
    n0 = int(v.shape[0]/sum(tile)*len(mask))
    if v.ndim == 1:
        v_full = np.zeros(n0)
    elif v.ndim == 2:
        v_full = np.zeros([n0, v.shape[1]])
    
    idxs = np.where(tile)[0].tolist()
    start_v = 0
    step_v = len(idxs)
    for i in idxs:
        start_v_full = i
        step_v_full = len(mask)
        v_full[start_v_full::step_v_full,] = v[start_v::step_v,]
        start_v += 1
        
    return v_full


def process(L, component):
    """
    Specify the default behavior of loading regarding the number of components L.
    """
    if not L:
        return num_basis[component]
    if L==0 or L=='all':
        return num_snapshots['train']

def save_parameters(parameters, dataset):
    path = join('dataset',dataset,'parameters')
    np.savetxt(path, parameters)

def load_parameters(dataset):
    path = join('dataset',dataset,'parameters')
    return np.loadtxt(path, parameters)

def save_snapshot(snapshot, dataset, index):
    path = join('dataset',dataset,f'truth_{index}')
    with open(path, 'wb+') as fid:
        snapshot.byteswap().tofile(fid)

def load_snapshot(dataset, i):
    file_path = join('dataset', dataset, f'truth_{i}')
    snapshot = np.fromfile(file_path, np.float64()).byteswap()
    return snapshot
    
def load_snapshots(dataset):
    t0 = time.time()
    S = []
    for i in range(num_snapshots[dataset]):
        snapshot = load_snapshot(dataset, i)
        S.append(snapshot)
    dt = time.time() - t0
    print(f'Loaded {num_snapshots[dataset]} {dataset} snapshots in {dt:.1f} s')
    return np.asarray(S).T

def save_npy_to_binary(numpy_in, path):
    numpy_in.byteswap().tofile(path)
    
def save_targets(dataset, component, targets):
    path = join('data','targets',F'{dataset}_targets_{component}')
    with open(path, 'wb+') as f:
        pickle.dump(targets, f)

def load_targets(dataset, component, L=None):
    path = join('data','targets',F'{dataset}_targets_{component}')
    with open(path, 'rb') as f:
        targets = pickle.load(f)
    L = process(L, component)
    return targets[:,:L] # shape N x L = #snapshots x #basis

def save_features(dataset, features):
    path = join('data','features',F'{dataset}_features')
    with open(path, 'wb+') as f:
        pickle.dump(features, f)
    
def load_features(dataset):
    path = join('data','features',F'{dataset}_features')
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_error_POD_sq(dataset, component, err_POD_sq):
    path = join('data','errors',F'{dataset}_error_POD_sq_{component}')
    with open(path, 'wb+') as f:
        pickle.dump(err_POD_sq, f)        

def load_error_POD_sq(dataset, component, L=None):
    path = join('data','errors',F'{dataset}_error_POD_sq_{component}')
    with open(path, 'rb') as f:
        err_POD_sq = pickle.load(f)
    L = process(L, component)
    # L+1 x N matrix containing squared standardized projection i.e. POD errors
    return err_POD_sq[:L+1,:]

def save_POD(V, D, S_mean, component):
    path_base = join('data','POD',component)
    ## S_mean
    path = join(path_base, 'S_mean')
    with open(path, 'wb+') as f:
        pickle.dump(S_mean, f)
    ## V
    path = join(path_base, 'V')
    with open(path, 'wb+') as f:
        pickle.dump(V, f)
    # D
    path = join(path_base, 'D')
    with open(path, 'wb+') as f:
        pickle.dump(D, f)

def load_POD_V(component, L=None):
    path = join('data','POD',component,'V')
    with open(path, 'rb') as f:
        V = pickle.load(f)
    L = process(L, component)
    return V[:,:L]

def load_POD_S_mean(component):
    path = join('data','POD',component,'S_mean')
    with open(path, 'rb') as f:
        S_mean = pickle.load(f)
    return S_mean

def load_POD_D_and_denom_sq(component, L=None):
    path = join('data','POD',component,'D')
    with open(path, 'rb') as f:
        D = pickle.load(f)
    denom_sq = np.sum(D**2)
    L = process(L, component)
    return D[:L], denom_sq

def save_scaler(scaler):
    path = join('data','scaler','scaler')
    with open(path, 'wb+') as f:
        pickle.dump(scaler, f)
    
def load_scaler():
    path = join('data','scaler','scaler')
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_error_table(df, dataset):
    path = join('data','errors',F'{dataset}_error_table')
    df.to_pickle(path)
    
def load_error_table(dataset):
    path = join('data','errors',F'{dataset}_error_table')
    return pd.read_pickle(path)

