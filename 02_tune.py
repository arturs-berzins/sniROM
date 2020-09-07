"""
Trains or tunes the models according to the specification. In the paper, we only
tune the FNN.
ray.tune start a parallel process for each configuration, so the tuning is
performed pretty quickly, if many processor threads are available. However,
modifying the tune_config can alleviate the computational burden.

Alternatively, the tuning can be disabled altogether by moving the 'FNN' key
from models_to_be_tuned to models_to_be_trained. This will automatically use a
sensible hyperparameters, namely, hidden_size=40 and lr=1e-2 which were found
close to optimal in the paper. However, the script 03_plot_tune.py will not
work.
"""
# Author: Arturs Berzins <berzins@cats.rwth-aachen.de>
# License: BSD 3 clause

import config
import utils
import numpy as np
from os.path import join
from ray import tune
import time

utils.set_up_model_directories()

models_to_be_trained = {'RBF','GPR'}
models_to_be_tuned   = {'FNN'}

# define the configuration for hyperparameter tuning
tune_config = {}
tune_config['FNN'] = {
 'lr':          tune.grid_search((10**np.linspace(-4,0,5)).tolist()),
 'hidden_size': tune.grid_search(np.linspace(10,50,9,dtype=int).tolist())
  }

def main():
    
    for component in config.components:
            
        D, denom_sq = utils.load_POD_D_and_denom_sq(component)
        
        features = {}
        targets = {}
        for dataset in ['train', 'validate']:
            features[dataset] = utils.load_features(dataset)
            targets[dataset]  = utils.load_targets(dataset, component)
        
        ## Wrapper for the training routine
        def train_wrapper(tune_config):
            model_constructor = utils.models[model_key]
            model = model_constructor()
            model.set_data(features, targets, D, denom_sq)
            model.train(tune_config)
            model.save(utils.model_dir, component)
        
        for model_key in models_to_be_trained:
            ## Train without a tuning config
            t0 = time.time()
            train_wrapper(None)
            dt = time.time() - t0
            print(F"Trained {model_key} for {component} in {dt:.4} s")
            
        for model_key in models_to_be_tuned:
            ## Tune using the defined tuning config
            analysis = tune.run(
                train_wrapper,
                local_dir= join(utils.model_dir, model_key),
                name=component,
                config=tune_config[model_key],
                stop={'time_total_s': 1800}
                )

if __name__ == '__main__':
    main()
