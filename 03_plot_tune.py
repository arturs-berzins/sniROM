"""
Plot the results of FNN tuning as in Figures 5, 11, 17.
"""
# Author: Arturs Berzins <berzins@cats.rwth-aachen.de>
# License: BSD 3 clause

import config
import utils
import os
import numpy as np
import pandas as pd
from ray import tune
import matplotlib
from matplotlib import pyplot, colors
from matplotlib.ticker import FixedLocator
import warnings

model_key = 'FNN'

preamble = (
    r'\usepackage{amsmath}'
    r'\usepackage{amssymb}'
    r'\newcommand{\vekt}[1]{\mbox{$\boldsymbol{#1}$}}'
    )

matplotlib.rcParams.update({
    'pgf.texsystem': "pdflatex",
    'font.family': 'serif',
    'text.latex.preamble': preamble,
    'pgf.preamble': preamble,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 8,
    'legend.fontsize': 8,
    'axes.labelsize':  8,
    'axes.titlesize':  8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

df_all = {}

for component in config.components:
    df_all[component] = pd.DataFrame()
    # https://ray.readthedocs.io/en/latest/tune-package-ref.html#ray.tune.Analysis
    # https://ray.readthedocs.io/en/latest/tune-usage.html#analyzing-results
    path_analysis = os.path.join(utils.model_dir, model_key, component)
    n = sum(f.endswith('.json') for f in os.listdir(path_analysis))
    if n>1:
        warnings.warn(F'{path_analysis} contains {n} experimental runs.'+
        'Results might not plot correctly, so you might want to remove previous runs.')
    analysis = tune.Analysis(path_analysis)
    df_temp = analysis.dataframe()
    df_all[component] = pd.concat([df_all[component],df_temp], ignore_index=True, sort=False)

### CONTOUR SENSITIVITY

fig, axes = pyplot.subplots(1, len(config.components), sharex=True, sharey=True)
fig.suptitle('')
legends = []
caxes = [None]*len(axes)

w=6.3
h=2.3
fig.set_size_inches(w=w, h=h)
fig.subplots_adjust(bottom=0, top=1, left=0, right=1, wspace=0)

for idx_ax, component in enumerate(config.components):
    df = df_all[component]
    lrs = df['config/lr']
    nhs = df['config/hidden_size']
    ls  = df['mean_loss']
    
    x = np.unique(lrs)
    y = np.unique(nhs)
    X, Y = np.meshgrid(x,y)
    Z = np.zeros(X.shape)
    for ix, vx in enumerate(x):
        for iy, vy in enumerate(y):
            # BEWARE: this can conflict with other experiments in the same folder
            z = df.loc[ (df['config/lr']==vx) &
                        (df['config/hidden_size']==vy)
                      ]['mean_loss'].values[0] ** 0.5
            Z[iy,ix] = z
    
    Z = np.clip(Z, None, 1)
    vmin = np.min(Z)
    vmax = np.max(Z)
    lev_exp = np.linspace(  np.log10(Z.min()), 
                            np.log10(Z.max()), 
                            20)
    levs = np.power(10, lev_exp)
    cf = axes[idx_ax].contourf(X, Y, Z, levs, norm=colors.LogNorm(), cmap='viridis')

    axes[idx_ax].set_xlabel(r'$\alpha_0$')
    axes[idx_ax].set_title(r'${}$'.format([r'\vekt{u}','p','T'][idx_ax]))
    
    axes[idx_ax].set_xscale('log')
    
    # mark best location 
    idx = df['mean_loss'].idxmin()
    lr_min = df.loc[idx]["config/lr"]
    nh_min = df.loc[idx]["config/hidden_size"]
    axes[idx_ax].scatter([lr_min], [nh_min], c='red',marker='+', linewidth=0.5)
    # force limits of axes because marker might create whitespace
    axes[idx_ax].set_xlim([lrs.min(), lrs.max()])
    axes[idx_ax].set_ylim([nhs.min(), nhs.max()])
    #axes[idx_ax].set_box_aspect(1)
    
    pos_ax = axes[idx_ax].get_position()
    #print(pos_ax)
    #print([pos_ax.width,pos_ax.height])
    margin_top = 0.1
    margin_bottom = 0.35
    scaler = (1-margin_top-margin_bottom)
    new_w = pos_ax.height*h/w*scaler
    new_h = pos_ax.height*scaler
    c=0.5
    wspace = c*new_w
    L = len(axes)
    left = (1-new_w*(L+(L-1)*c))/2
    new_x0 = left + idx_ax*(wspace+new_w)
    new_y0 = margin_bottom
    pos_ax = [new_x0, new_y0, new_w, new_h]
    axes[idx_ax].set_position(pos_ax)

    #divider = make_axes_locatable(axes[idx_ax])
    #caxes[idx_ax] = divider.append_axes('bottom', size='10%', pad=0)
    caxes[idx_ax] = pyplot.axes([new_x0, 0.05, new_w, 0.1*new_h])
    #cax.set_title(r'$\varepsilon_{FNN}$', size='medium', position='bottom')
    bar = fig.colorbar(cf, cax=caxes[idx_ax], orientation='horizontal')
    
    exp_min = np.log10(Z.min())
    exp_max = np.log10(Z.max())
    
    # MAJOR
    exps_major = np.arange( np.floor(exp_min), np.ceil(exp_max) +1)
    levs_major = 10**exps_major
    # linearly interpolate exponent: exp_min-exp_max -> 0-1 
    #locs_major = (exps_major - exp_min) / (exp_max-exp_min)
    caxes[idx_ax].xaxis.set_major_locator(FixedLocator(levs_major))
    # NOTE: FixedLocator in old matplotlib versions expected locations in terms of axis
    # newer versions seem to use values per default, so we don't have to compute levels
    
    # MINOR
    levs_minor = []
    # Add minor tick levels ..
    # .. below first major level
    for f in range(2,9+1):
        levs_minor.append(levs_major[0]*0.1*f)
    # .. above each major level
    for lev_major in levs_major:
        for f in range(2,9+1):
            levs_minor.append(lev_major*f)
            
    caxes[idx_ax].xaxis.set_minor_locator(FixedLocator(levs_minor))
    
    m, e = '{:.2e}'.format(Z.min()).split('e')
    label_min = r'${}{{\times}}10^{{{}}}$'.format(m,int(e))
    if Z.max() == 1:
        label_max = r'$\ge 1$'
    else:
        m, e = '{:.2e}'.format(Z.max()).split('e')
        label_max = r'${}{{\times}}10^{{{}}}$'.format(m,int(e))
    
    caxes[idx_ax].annotate(label_min, xy=(0, 1.2), xytext=(0, 0),
                xycoords='axes fraction', textcoords='offset points',
                ha='left', va='bottom')
    caxes[idx_ax].annotate(label_max, xy=(1, 1.2), xytext=(0, 0),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='bottom')
                # va basline or top seems to work best
    
axes[0].set_ylabel(r'$N_\nu$')

pyplot.show()
