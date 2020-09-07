"""
Make predictions and store those on the disk for visualization (or further
evaluation). Currently, this script uses the test_Hesthaven_Ubbiali set as in
Figure 8. For different problems the dataset and idxs of interest must be
specified accordingly.
"""
# Author: Arturs Berzins <berzins@cats.rwth-aachen.de>
# License: BSD 3 clause

import config
import utils
import numpy as np
import pandas as pd
import os
join = os.path.join

model_key = 'FNN'
dataset = 'test_Hesthaven_Ubbiali'
idxs = [0, 1, 2]

features = utils.load_features(dataset)
features = features[idxs]

predictions = {}    # dict
projections = {}    # dict
truths      = {}    # dict
error_table = []    # list of dicts

for component in config.components:

    V = utils.load_POD_V(component)
    S_mean = utils.load_POD_S_mean(component)
    D, denom_sq = utils.load_POD_D_and_denom_sq(component)

    model_constructor = utils.models[model_key]
    model = model_constructor()
    model.load(utils.model_dir, component)
    outputs = model.evaluate(features)

    for i, idx in enumerate(idxs):
        ## Infer reduced coefficients
        y_s = outputs[i]
        ## Load truth
        snapshot = utils.load_snapshot(dataset, idx)
        ## Mask other components
        solution_TRUE = utils.reduce(snapshot, component)
        ## Project coefficients back to full solution space (Eq. 
        solution_PRED = S_mean + np.matmul(V, np.matmul(np.diag(D), y_s))
        solution_PROJ = S_mean + np.matmul(V, np.matmul(V.T, solution_TRUE-S_mean))
         
        eps_pod     = ( ((solution_TRUE - solution_PROJ)**2).sum() / denom_sq ) ** .5
        eps_reg     = ( ((solution_PRED - solution_PROJ)**2).sum() / denom_sq ) ** .5
        eps_podreg  = ( ((solution_TRUE - solution_PRED)**2).sum() / denom_sq ) ** .5
        #print(f'Sanity check for {idx}: \t{eps_pod}^2 + \t{eps_reg}^2 =
        #   \t{eps_pod**2+eps_reg**2} \t vs \t{eps_podreg**2}={eps_podreg}^2')
        
        if idx in predictions:
            predictions[idx] += utils.expand(solution_PRED, component)
            projections[idx] += utils.expand(solution_PROJ, component)
            truths[idx]      += utils.expand(solution_TRUE, component)
        else:
            predictions[idx] = utils.expand(solution_PRED, component)
            projections[idx] = utils.expand(solution_PROJ, component)
            truths[idx]      = utils.expand(solution_TRUE, component)
        
        error_table.append({'component':  component,
                            'sample':     idx,
                            'eps_pod':    eps_pod,
                            'eps_podreg': eps_podreg
                           })

## Store and print errors
path_root = join('visualization', dataset)
os.makedirs(path_root, exist_ok=True)
df = pd.DataFrame(error_table)
path_errors = join(path_root, 'errors.csv')
df.to_csv(path_errors)
print(df)

## Store results on disk
for idx in idxs:
    utils.save_npy_to_binary(predictions[idx], join(path_root, F'pred_{idx}'))
    utils.save_npy_to_binary(projections[idx], join(path_root, F'proj_{idx}'))
    utils.save_npy_to_binary(truths[idx],      join(path_root, F'truth_{idx}'))
    delta = truths[idx]-predictions[idx]
    utils.save_npy_to_binary(delta,            join(path_root, F'delta_{idx}'))
