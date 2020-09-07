"""
Model for Gauss process regression (GPR).
"""
# Author: Arturs Berzins <berzins@cats.rwth-aachen.de>
# License: BSD 3 clause

import numpy as np
import pickle
from sklearn.multioutput import MultiOutputRegressor
#from sklearn.gaussian_process import GaussianProcessRegressor
from models.GPR.my_gpr import myGaussianProcessRegressor as GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import os

class GPR:
    def __init__(self):
        pass
        
    def set_data(self, features, targets, D, denom_sq):
        self.features = features
        self.targets = targets
        self.D = D
        self.inv_denom_sq = denom_sq**-1
    
    def train(self, config):
        input_size = self.features['train'].shape[1]
        
        alpha = 1e-9# 1e-5
        # IMPORTANT: if no kernel is specified, a constant one will be used per default.
        # The constant kernels hyperparameters will NOT be optimized!
        #kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kernel = 0.01 * RBF(length_scale=[0.1]*input_size, length_scale_bounds=(1e-2, 1e+2)) \
                + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-10, 1e0))
        regressor = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=10)
        self.model = MultiOutputRegressor(regressor)
        
        self.model.fit(self.features['train'], self.targets['train'])
        # Print learnt hyperparameters
        #for e in self.model.estimators_:
        #    print(e.kernel_.get_params())
        
    def evaluate(self, features):
        return self.model.predict(features)
   
    def test(self):
        f = self.features['test']
        t = self.targets['test']
        q_rb = self.evaluate(f)
        eps_reg_sq = np.sum((self.D*(q_rb - t))**2) * self.inv_denom_sq / f.shape[0]
        return eps_reg_sq ** 0.5
    
    def save(self, model_dir, component):
        path = os.path.join(model_dir,'GPR',component,'model')
        with open(path, 'wb+') as f:
            pickle.dump(self.model, f)
    
    def load(self, model_dir, component):
        path = os.path.join(model_dir,'GPR',component,'model')
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
