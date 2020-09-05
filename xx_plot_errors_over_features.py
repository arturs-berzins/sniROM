"""
Plot standardized errors over features for all components.
All features are standardized.
Observe the behavior of model and projection errors over the parameter space.
The errors are likely to be worse at the extremes of the space.
"""
import config
import utils
import numpy as np
from matplotlib import pyplot

dataset = 'test'
model_keys = ['RBF',
              'GPR',
              'FNN',
              ''   # projection error
              ]
df = utils.load_error_table(dataset)

P = len(config.mu_names)
fig, axes = pyplot.subplots(len(config.components),P,sharex='col',sharey='row')
fig.suptitle('Standardized errors over features')
cmap = pyplot.get_cmap('tab10')

features = utils.load_features(dataset)

for idx_ax, component in enumerate(config.components):
    L = config.num_basis[component]
    df_filtered = df.loc[ (df['component']==component) &
                          (df['l']==L)]
    
    for p in range(P):
        xs = features[:,p]
        for i, model_key in enumerate(model_keys):
            ys = df_filtered[F'eps_pod{model_key.lower()}_sq'].values
            ys = ys ** 0.5
            # Plot transparent data due to bug in scatter limits
            axes[idx_ax,p].plot(xs, ys, alpha=0)
            axes[idx_ax,p].scatter(xs, ys, marker='o', s=2, color=cmap(i), label=F'POD-{model_key}')

# Label x axes
for p in range(P):
    axes[len(config.components)-1,p].set_xlabel(config.mu_names[p])
# Label y axes
for idx_ax, component in enumerate(config.components):
    axes[idx_ax,0].set_ylabel(F'{component}')

handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(model_keys))

fig.set_size_inches(w=6.3, h=4.3)    
fig.subplots_adjust(bottom=.2)

pyplot.show()
