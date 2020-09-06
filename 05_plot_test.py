"""
Plot the performance on the test set as in Figures 6, 12, 18.
"""

import config
import utils
import matplotlib
from matplotlib import pyplot
import numpy as np
import pandas as pd
from matplotlib import ticker, cm, colors
from matplotlib.ticker import FormatStrFormatter, LogLocator, FixedLocator, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

preamble = (
    r'\usepackage{amsmath}'
    r'\usepackage{amssymb}'
    r'\newcommand{\vekt}[1]{\mbox{$\boldsymbol{#1}$}}'
    )
            
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': preamble,
    'pgf.preamble': preamble,
    'pgf.rcfonts': False,
    'font.size': 8
})
#https://timodenk.com/blog/exporting-matplotlib-plots-to-latex/

## In case we don't want to plot all L for visibility
L_max = [20,20,20]
dataset = 'test'
df = utils.load_error_table(dataset)
fig, axes = pyplot.subplots(1,len(config.components))
fig.suptitle('')
legends = []
cmap = pyplot.get_cmap('tab10')
keys = [F'eps_pod{m.lower()}_sq' for m in list(utils.models.keys()) + ['']]

for idx_ax, component in enumerate(config.components):
    L = config.num_basis[component]
    ls = [l for l in range(L+1)]
    
    df_filtered = df.loc[ (df['component']==component)]
    
    markers = ['1','2','3','4']
    
    for i, key in enumerate(keys):
        mean_sq = np.array([ df_filtered.loc[(df_filtered['l']==l)][key].mean() for l in ls ])
        mean    = np.array([(df_filtered.loc[(df_filtered['l']==l)][key]**0.5).mean() for l in ls ])
        
        rmse = np.sqrt(mean_sq)
        axes[idx_ax].plot(ls, rmse, color=cmap(i), marker=markers[i], markersize=6, linewidth=1, markevery=5)
    
    axes[idx_ax].set_yscale('log')
    axes[idx_ax].set_ylim([None,1])
    axes[idx_ax].set_xlim([0, L_max[idx_ax]])
    #axes[idx_ax].set_xlabel(r'$L_{}$'.format(component))
    axes[idx_ax].set_xlabel(r'$L$')
    axes[idx_ax].grid(which='major', axis='y', linestyle=':')
    axes[idx_ax].grid(which='major', axis='x', linestyle=':')
    axes[idx_ax].title.set_text(r'${}$'.format([r'\vekt{u}',r'p',r'T'][idx_ax]))
    
    # all 10 integer powers between min and max
    displayed_data = rmse[:L_max[idx_ax]]
    exp_min = np.floor(np.log10(np.min(displayed_data)))
    exp_max = 0#np.log10(np.max(rmse))
    axes[idx_ax].set_ylim([10**exp_min, 10**exp_max])
    # MAJOR
    exps_major = np.arange( exp_min, exp_max+1 )
    axes[idx_ax].yaxis.set_major_locator(FixedLocator(10**exps_major))
    
    # MINOR
    axes[idx_ax].yaxis.set_minor_formatter(NullFormatter())
    
    axes[idx_ax].yaxis.set_ticks_position('both')
    axes[idx_ax].tick_params(axis='y', direction="in", which='both')

labels = [r'${\varepsilon}' + r'_\text{{{}}}'.format(key) + r'(\mathbb{P}_{te})$' for key in [r'POD-RBF', 'POD-GPR', 'POD-ANN', 'POD'] ]
lines = axes[0].get_lines()
fig.legend(lines, labels, ncol = 4, loc="lower center")
fig.subplots_adjust(bottom=0.35, top=0.90, left=0.10, right=0.95, wspace=0.35, hspace=0.20)

fig.set_size_inches(w=6.3, h=2.3)
     
pyplot.show()
