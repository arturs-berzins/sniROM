"""
Converts the dataset into features and targets for the regression models.
Stores these, as well as the feature scaler, POD matrices and projection errors
under /data. See Section 2.
"""
import config
import utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import time

def main():

    utils.set_up_data_directories()
    
    snapshots = {}
    parameters = {}
    for dataset in config.datasets:
        # shape: N_h x N
        # i.e. #DOFs x #snapshots
        snapshots[dataset]  = utils.load_snapshots(dataset) 
        parameters[dataset] = utils.load_parameters(dataset)
    
    for component in config.components:
        assert config.datasets[0] == 'train', 'The first dataset must be train'
        print(f'\nComputing targets for component {component}')
        
        for dataset in config.datasets:
            # Snapshot matrix, non-centered
            S_n = utils.reduce(snapshots[dataset], component)
                        
            if dataset == 'train':
                # Compute and store ..
                # .. mean and POD
                S_mean = np.mean(S_n,axis=1)
                S = np.array([col - S_mean for col in S_n.T]).T
                V, D = do_POD(S)
                utils.save_POD(V, D, S_mean, component)
                # .. scaler
                scaler = StandardScaler()
                scaler.fit(parameters[dataset])
                utils.save_scaler(scaler)
            else:
                # Compute centered snapshot matrix
                S = np.array([col - S_mean for col in S_n.T]).T   
            
            # Now V, D, S_mean and scaler are available
            
            # Compute and store ..                 
            # .. features
            features = compute_features(scaler, parameters[dataset])
            utils.save_features(dataset, features)
            # .. targets
            targets = compute_targets(S, V, D) 
            utils.save_targets(dataset, component, targets)
            # .. projection error
            err_POD_sq = compute_error_POD_sq(S, V, D)
            utils.save_error_POD_sq(dataset, component, err_POD_sq)

def compute_features(scaler, parameters):
    return scaler.transform(parameters)

def compute_targets(S, V, D):
    return np.matmul(np.diag(D**-1),np.matmul(V.T, S)).T

def compute_error_POD_sq(S, V, D):
    L = V.shape[1]  # max number of basis functions
    N = S.shape[1]  # number of snapshots in snapshot matrix
    
    err_POD_sq = np.zeros([L+1, N])
    
    # precompute things for speed, because naive approach is extremely slow
    VTSsq = np.matmul(V.T, S) ** 2
    q_norms_sq = (S**2).sum(axis=0)
    inv_denom_sq = np.sum(D**2)**-1
    
    for l in range(L+1):
        err_POD_sq[l] = (q_norms_sq - VTSsq[:l,:].sum(axis=0)) * inv_denom_sq
    
    return err_POD_sq


def do_POD(S):
    t0 = time.time()
    
    N   = S.shape[1]      # number of snapshots
    N_h = S.shape[0]      # size of each snapshot
    L = N                 # compute and return the full decomposition
    print(f'N_h={N_h}, N={N}')
    if(N_h <= N):
        print('Matrix is wide, computing directly')
        M = np.matmul(S, S.T)/N                # compute the row covariance matrix
        eigenvalues, eigenvectors = eigh(M)    # compute eigenvalue decomposition
        eigenvectors = eigenvectors.real       # keep only the real part
        eigenvalues = eigenvalues.real         # keep only the real part
        idx = eigenvalues.argsort()[::-1]      # indices to sort in decreasing order
        eigenvalues = eigenvalues[idx]         # sort eigenvalues in decreasing order
        eigenvectors = eigenvectors[:, idx]    # sort eigenvectors correspondigly
        sig = abs(eigenvalues)**0.5
        #TODO: deal with numerically negative eigenvalues. Truncate to 0?
        
        V = eigenvectors[:,0:L]         # Reduced basis matrix from left singular vectors
        D = sig[0:L]                    # Singular values
        
    else:
        print('Matrix is tall, computing via covariance')
        M = np.matmul(S.T, S)/N                # compute the covariance matrix
        eigenvalues, eigenvectors = eigh(M)    # compute eigenvalue decomposition
        eigenvectors = eigenvectors.real       # keep only the real part
        eigenvalues = eigenvalues.real         # keep only the real part
        idx = eigenvalues.argsort()[::-1]      # indices to sort in decreasing order
        eigenvalues = eigenvalues[idx]         # sort eigenvalues in decreasing order
        eigenvectors = eigenvectors[:, idx]    # sort eigenvectors correspondigly
        
        W = np.array([(abs(eigenvalues[l]) ** -0.5 * np.matmul(S/(N**0.5), eigenvectors[:,l])).T for l in range(L)]).T
        sig = abs(eigenvalues)**0.5   
        
        V = W[:,0:L]                    # Reduced basis matrix from left singular vectors
        D = sig[0:L]                    # Singular values
        # TODO: rewrite in same notation as in paper
    
    dt = time.time() - t0
    print(f'Computed POD in {dt:.4} s')
    
    return V, D

if __name__ == '__main__':
    main()
