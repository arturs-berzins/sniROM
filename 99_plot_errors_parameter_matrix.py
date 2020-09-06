"""
Plot standardized errors over features for all components.
All features are standardized.
Errors are represented by areas of the circles.
Observe the behavior of model and projection errors over the parameter space.
The errors are likely to be worse at the extremes of the space.
"""
import config
import utils
import numpy as np
from matplotlib import pyplot

dataset = 'test'
model_key = 'FNN'
df = utils.load_error_table(dataset)
features = utils.load_features(dataset)
P = len(config.mu_names)
error_key = F'eps_pod{model_key.lower()}_sq'

for component in config.components:
    fig, axes = pyplot.subplots(P,P,sharex='col',sharey='row')
    fig.suptitle(f'Standardized POD-{model_key} errors represented by area for component {component}')

    L = config.num_basis[component]
    df_filtered = df.loc[ (df['component']==component) &
                          (df['l']==L)]

    # Plot matrix
    for p1 in range(P):
        for p2 in range(P):
            xs = features[:,p2]
            ys = features[:,p1]
            cs = df_filtered[error_key].values
            cs = cs ** 0.5
            # Plot transparent data due to bug in scatter limits
            axes[p1,p2].plot(xs, ys, alpha=0)
            # The scale factor is empirical, adjust for better visuals perception
            axes[p1,p2].scatter(xs, ys, s=(1e2*cs))
    
    # Label x axes
    for p in range(P):
        axes[P-1,p].set_xlabel(config.mu_names[p])
    # Label y axes
    for p in range(P):
        axes[p,0].set_ylabel(config.mu_names[p])

    fig.set_size_inches(w=6.3, h=6.3)     
pyplot.show()
