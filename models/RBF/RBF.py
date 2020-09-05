import numpy as np
import pickle
import scipy.interpolate
from ray import tune
import os

class RBF:
    def __init__(self):
        pass
    
    def set_data(self, features, targets, D, denom_sq):
        self.features = features
        self.targets = targets
        self.D = D
        self.inv_denom_sq = denom_sq**-1
    
    def train(self, config):
        self.model = scipy.interpolate.Rbf(*self.features['train'].T, self.targets['train'], function='multiquadric', mode='N-D')
    
    def evaluate(self, features):
        return self.model(*features.T)
    
    def test(self):
        f = self.features['test']
        t = self.targets['test']
        q_rb = self.evaluate(f)
        eps_reg_sq = np.sum((self.D*(q_rb - t))**2) * self.inv_denom_sq / f.shape[0]
        return eps_reg_sq ** 0.5
    
    
    def save(self, model_dir, component):
        path = os.path.join(model_dir,'RBF',component,'model')
        with open(path, 'wb+') as f:
            pickle.dump(self.model, f)
    
    def load(self, model_dir, component):
        path = os.path.join(model_dir,'RBF',component,'model')
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
