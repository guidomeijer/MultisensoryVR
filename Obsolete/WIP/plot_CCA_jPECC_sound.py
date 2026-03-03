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
#TICKS = [-1, 0, 1, 2]
TICKS = [-0.5, 0, 0.5]

# Load in data
path_dict = paths()
cca_df = pd.read_pickle(join(path_dict['save_path'], 'jPECC_sound_10ms-bins.pickle'))
time_ax = cca_df['time'][0]

# Plot
f, axs = plt.subplots(1, np.unique(cca_df['region_pair']).shape[0], sharey=True, sharex=True,
                      figsize=(1.75*np.unique(cca_df['region_pair']).shape[0], 5.25), dpi=dpi)
for i, region_pair in enumerate(np.unique(cca_df['region_pair'])):
    
    # Get mean jPECC for goal and distractor object entries
    #r_goal = np.dstack(cca_df.loc[(cca_df['region_pair'] == region_pair) & (cca_df['goal'] == 1) , 'r'])
    #r_goal = np.mean(r_goal, axis=2)
    r_sound = cca_df.loc[cca_df['region_pair'] == region_pair , 'r'].values[0]
        
    # Plot
    axs[i].imshow(np.flipud(r_sound), vmin=CLIM[0], vmax=CLIM[1],
                  cmap=CMAP,
                  extent=[time_ax[0], time_ax[-1], time_ax[0], time_ax[-1]],
                  interpolation=None)
    axs[i].set(title=f'{region_pair}', yticks=TICKS, xticks=TICKS,
               xlim=[np.round(time_ax[0], decimals=1), np.round(time_ax[-1], decimals=1)],
               ylim=[np.round(time_ax[0], decimals=1), np.round(time_ax[-1], decimals=1)])
        
   
    