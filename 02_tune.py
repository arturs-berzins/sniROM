import config
import utils
import numpy as np
import torch
import os
from ray import tune

utils.set_up_model_directories()

models_to_be_trained = {'RBF','GPR'}
models_to_be_tuned   = {'FNN'}

# define the configuration for hyperparameter tuning based on Ray Tune
tune_config = {}
tune_config['FNN'] = { 
                      'lr':             tune.grid_search((10**np.linspace(-4,0,3)).tolist()),
                      'hidden_size':    tune.grid_search(np.linspace(10,50,3,dtype=int).tolist())
                      }
assert set(tune_config.keys()) == set(models_to_be_tuned), "tune_config is incomplete"

def main():
    
    for component in config.components:
            
        D, denom_sq = utils.load_POD_D_and_denom_sq(component)
        
        features = {}
        targets = {}
        for dataset in ['train', 'validate']:
            features[dataset] = utils.load_features(dataset)
            targets[dataset]  = utils.load_targets(dataset, component)
        
        input_size  = features['train'].shape[1] # = len(config.mu_range)
        output_size =  targets['train'].shape[1] # = config.num_basis[component]
        
        
        ## Wrapper for the training routine
        def train_wrapper(config):
            model_constructor = utils.models[model_key]
            model = model_constructor()
            model.set_data(features, targets, D, denom_sq)
            model.train(config)
            model.save(utils.model_dir, component)
        
        for model_key in models_to_be_trained:
            ## Train without a tuning config
            train_wrapper(None)
            
        for model_key in models_to_be_tuned:
            ## Tune using the defined tuning config
            analysis = tune.run(
                train_wrapper,
                local_dir= os.path.join(utils.model_dir, model_key),
                name=component,
                config=tune_config[model_key],
                stop={'time_total_s': 1800}
                )

if __name__ == '__main__':
    main()
