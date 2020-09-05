import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch import autograd
from models.FNN import data_loader
from models.FNN.model import Model
import os
import pickle
from ray import tune
torch.manual_seed(0)

class FNN: 
    def __init__(self):
        ## Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def set_data(self, features, targets, D, denom_sq):
        self.features_np = features
        self.targets_np = targets
        self.D_np = D
        self.inv_denom_sq = denom_sq**-1
    
    def train(self, config):
        ## Internal config
        self.config = {}
        self.config['num_epochs']       = 500
        self.config['n_hidden']         = 2
        self.config['hidden_size']      = 20
        self.config['batch_size']       = 10
        self.config['lr']               = 0.01
        self.config['regularization']   = 1e-10
        # Overwrite internal config values given in the external config
        if config:
            for key in config.keys():
                self.config[key] = config[key]
        
        ## Model
        self.config['input_size'] = self.features_np['train'].shape[1]
        self.config['output_size'] = self.targets_np['train'].shape[1]
        self.model = Model(self.config).to(self.device)
        
        ## Data loaders
        self.batch_size = self.config['batch_size']
        self.train_loader = data_loader.create_loader(
            self.features_np['train'],
            self.targets_np['train'],
            self.batch_size,
            True)
        self.validate_loader  = data_loader.create_loader(
            self.features_np['validate'],
            self.targets_np['validate'],
            self.features_np['validate'].shape[0], # use all test samples
            False)                             # don't shuffle
        
        ## Hyperparameters
        self.num_epochs = self.config['num_epochs']
        self.learning_rate = self.config['lr']
        
        ## Loss and optimizer
        self.criterion = self.eps_reg_sq
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-8, weight_decay=self.config['regularization'])
        lambdaLR = lambda epoch: 1 / (1 + 0.005*epoch)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambdaLR)
        
        self.train_start()
    
    def train_start(self):
        ## Train
        early_stop = False
        self.D = torch.from_numpy(self.D_np).float().to(self.device)
        
        for epoch in range(self.num_epochs):
            for i, (features, targets) in enumerate(self.train_loader):
                self.model.train()
                self.optimizer.zero_grad()
                
                # Move tensors to the configured device
                features = features.to(self.device)
                targets  =  targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets) ** 0.5
                if torch.isnan(loss):
                    print('Something went nan, stopping')
                    early_stop = True
                    break # break out of this batch

                # Backward and optimize
                loss.backward()
                self.optimizer.step()
            
            if early_stop:
                break # break out of this epoch
                
            self.scheduler.step()
                
            if epoch%10==0 or epoch==self.num_epochs-1:
                validate_loss  = self.get_loss(self.validate_loader)
                train_loss = self.get_loss(self.train_loader)
                print('eps_reg: Epoch [{}/{}], LR: {:.2e}, Train loss: {:.2e}, Validate loss: {:.2e}'
                    .format(epoch+1, self.num_epochs, self.scheduler.get_lr()[0], train_loss.item()**0.5, validate_loss.item()**0.5))
                
                tune.track.log(mean_loss = validate_loss.item(), episodes_this_iter = 10)
                    
        return self

    def eps_reg_sq(self, outputs, targets):
        return torch.sum((self.D*(targets - outputs)) ** 2) * self.inv_denom_sq / targets.shape[0]
        
    def get_loss(self, loader):
        with torch.no_grad():
            self.model.eval()
            loss = 0.0
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(features)
                loss += self.criterion(outputs, targets)
            return loss/len(loader)

    def evaluate(self, features):
        with torch.no_grad():
            self.model.eval()
            output = self.model(torch.tensor(features).float())
            u_rb = output.numpy()
            return u_rb
    
    def save(self, model_dir, component):
        path_config     = os.path.join(tune.track.trial_dir(),'config')
        path_state_dict = os.path.join(tune.track.trial_dir(),'state_dict')
        
        with open(path_config, 'wb+') as f:
            pickle.dump(self.config, f)
        
        torch.save(self.model.state_dict(), path_state_dict)
    
    def load(self, model_dir, component):
        '''
        Find and loads the best model from ray.tune analysis results.
        '''
        path_analysis = os.path.join(model_dir,'FNN',component)
        analysis = tune.Analysis(path_analysis)
        df_temp = analysis.dataframe()
        idx = df_temp['mean_loss'].idxmin()
        logdir = df_temp.loc[idx]['logdir']
        
        path_config     = os.path.join(logdir,'config')
        path_state_dict = os.path.join(logdir,'state_dict')
        
        with open(path_config, 'rb') as f:
            config = pickle.load(f)
            self.model = Model(config).to(self.device)
        
        state_dict = torch.load(path_state_dict,
                                map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
