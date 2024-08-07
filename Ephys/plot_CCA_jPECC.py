# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:27:07 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style
colors, dpi = figure_style()

# Settings
CLIM = [-1, 1]
#CMAP = sns.diverging_palette(250, 15, center="dark", as_cmap=True)
CMAP = 'coolwarm'
TICKS = [-1, 0, 1, 2]

# Load in data
path_dict = paths()
cca_df = pd.read_pickle(join(path_dict['save_path'], 'jPECC_goal_10ms-bins.pickle'))
time_ax = cca_df['time'][0]

# Plot
f, axs = plt.subplots(3, np.unique(cca_df['region_pair']).shape[0], sharey=True, sharex=True,
                      figsize=(1.75*np.unique(cca_df['region_pair']).shape[0], 5.25), dpi=dpi)
for i, region_pair in enumerate(np.unique(cca_df['region_pair'])):
    
    # Get mean jPECC for goal and distractor object entries
    r_goal = np.dstack(cca_df.loc[(cca_df['region_pair'] == region_pair) & (cca_df['goal'] == 1) , 'r'])
    r_goal = np.mean(r_goal, axis=2)
    r_dis = np.dstack(cca_df.loc[(cca_df['region_pair'] == region_pair) & (cca_df['goal'] == 0) , 'r'])
    r_dis = np.mean(r_dis, axis=2)
    r_diff = r_goal - r_dis
    
    # Plot
    axs[0, i].imshow(np.flipud(r_goal), clim=CLIM, vmin=CLIM[0], vmax=CLIM[1],
                     cmap=CMAP,
                     extent=[time_ax[0], time_ax[-1], time_ax[0], time_ax[-1]],
                     interpolation=None)
    axs[0, i].set(title=f'{region_pair}', yticks=TICKS, xticks=TICKS,
                  xlim=[np.round(time_ax[0]), np.round(time_ax[-1])],
                  ylim=[np.round(time_ax[0]), np.round(time_ax[-1])])
        
    axs[1, i].imshow(np.flipud(r_dis), clim=CLIM,  vmin=CLIM[0], vmax=CLIM[1],
                     cmap=CMAP,
                     extent=[time_ax[0], time_ax[-1], time_ax[0], time_ax[-1]],
                     interpolation=None)
    
    axs[2, i].imshow(np.flipud(r_diff), clim=[-0.7, 0.7], cmap=CMAP,
                     extent=[time_ax[0], time_ax[-1], time_ax[0], time_ax[-1]],
                     interpolation=None)
    # 

    