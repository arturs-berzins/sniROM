"""
Use the trained models to evaluate the performance on the previously unseed test
dataset.
Loading the FNN automatically detects the best hyperparameter configuration from
the tuning.
"""
import config
import utils
import numpy as np
import os
import pandas as pd

def main():
    dataset = 'test'
    features = utils.load_features(dataset)

    error_table = [] #list of dictionaries
        
    for component in config.components:
        L = config.num_basis[component]
        D, denom_sq = utils.load_POD_D_and_denom_sq(component)
        eps_pod_sqs = utils.load_error_POD_sq(dataset, component)
        
        targets = utils.load_targets(dataset, component)
        outputs = {}
        for model_key, model_constructor in utils.models.items():
            model = model_constructor()
            model.load(utils.model_dir, component)
            outputs[model_key] = model.evaluate(features)

        ## for each sample in test set
        for i in range(len(targets)):
            
            for l in range(L+1):
                eps_pod_sq = eps_pod_sqs[l,i]
                line = {'component': component,
                        'l': l,
                        'sample': i,
                        'dataset': dataset,
                        'eps_pod_sq': eps_pod_sq
                       }
                
                for model_key in utils.models:
                    q_rb_model = outputs[model_key][i][:l]
                    q_rb_truth = targets[i][:l]
                    D_l = D[:l]
                    eps_reg_sq = np.sum((D_l*(q_rb_model-q_rb_truth))**2) / denom_sq
                    eps_sq = eps_pod_sq + eps_reg_sq
                    eps_key = F'eps_pod{model_key.lower()}_sq'
                    line[eps_key] = eps_sq
                
                error_table.append(line)
              
    df = pd.DataFrame(error_table)
    utils.save_error_table(df, dataset)

if __name__ == '__main__':
    main()
