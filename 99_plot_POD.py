"""
Plot squared aggregate standardized projection error over number of bases L.
Observe, how rapidly the error decreases over L. Compare, how close the
behaviour of the other datasets is. Dissimilar behaviour- the datasets do not
cover the sample space well enough i.e. can be seen as not from the same
distribution. See Eq. (17), (22)-(24) in paper.
"""
# Author: Arturs Berzins <berzins@cats.rwth-aachen.de>
# License: BSD 3 clause

import config
import utils
from matplotlib import pyplot


fig, axes = pyplot.subplots(1,len(config.components))
fig.suptitle('Squared projection loss over number of basis functions')

for idx_ax, component in enumerate(config.components):
    L = config.num_basis[component]
    
    for dataset in config.datasets:
        eps_pod_sq = utils.load_error_POD_sq(dataset, component, L='all')
        axes[idx_ax].plot(eps_pod_sq.mean(axis=1)**1, label=f'{dataset}')
       
    axes[idx_ax].set_yscale('log')
    axes[idx_ax].set_xlabel('L')
    axes[idx_ax].legend()    
 
axes[0].set_ylabel(r'$\varepsilon^2_{POD}$')
fig.set_size_inches(w=6.3, h=4.0)
pyplot.show()
