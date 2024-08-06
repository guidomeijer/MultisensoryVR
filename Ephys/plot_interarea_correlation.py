# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:05:33 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style
colors, dpi = figure_style()

# Load in data
path_dict = paths()
corr_df = pd.read_csv(join(path_dict['save_path'], 'region_corr_50ms-bins.csv'))

# Plot
region_pairs = np.unique(corr_df['region_pair'])
f, axs = plt.subplots(1, len(region_pairs), figsize=(1.75*len(region_pairs), 1.75), dpi=dpi)
for i, region_pair in enumerate(region_pairs):
    long_df = pd.melt(corr_df[corr_df['region_pair'] == region_pair],
                      id_vars='time', value_vars=['r_goal_baseline', 'r_distractor_baseline'])
    sns.lineplot(data=long_df, x='time', y='value', hue='variable', ax=axs[i], errorbar='se',
                 legend=None, hue_order=['r_goal_baseline', 'r_distractor_baseline'],
                 palette=[colors['goal'], colors['no-goal']])
    axs[i].set(title=f'{region_pair}')
