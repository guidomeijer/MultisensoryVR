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
corr_df = pd.read_csv(join(path_dict['google_drive_data_path'], 'region_corr_100ms-bins.csv'))

# Standardize region pairs to group reciprocal pairs (e.g., TEa-CA1 and CA1-TEa) together
corr_df['region_pair'] = np.where(corr_df['region_1'] < corr_df['region_2'],
                                  corr_df['region_1'] + '-' + corr_df['region_2'],
                                  corr_df['region_2'] + '-' + corr_df['region_1'])

# Plot
region_pairs = np.unique(corr_df['region_pair'])
f, axs = plt.subplots(3, 5, figsize=(8, 5), dpi=dpi, sharex=True)
axs = axs.flatten()
for i, region_pair in enumerate(region_pairs):

    this_df = corr_df[corr_df['region_pair'] == region_pair]
    sns.lineplot(this_df, x='time', y='r_obj1_rew_baseline', color=colors['goal'], errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    sns.lineplot(this_df, x='time', y='r_obj1_no_rew_baseline', color=colors['no-goal'], errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    axs[i].set(title=f'{region_pair}', ylabel='', xlabel='')
    
axs[0].set(ylabel='Correlation (r)')
sns.despine(trim=True)
plt.tight_layout()

f, axs = plt.subplots(3, 5, figsize=(8, 5), dpi=dpi, sharex=True)
axs = axs.flatten()
for i, region_pair in enumerate(region_pairs):
    this_df = corr_df[corr_df['region_pair'] == region_pair]
    sns.lineplot(this_df, x='time', y='r_obj2_rew_baseline', color=colors['goal'], errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    sns.lineplot(this_df, x='time', y='r_obj2_no_rew_baseline', color=colors['no-goal'], errorbar='se', err_kws={'lw': 0}, ax=axs[i])
    axs[i].set(title=f'{region_pair}', ylabel='', xlabel='')

axs[0].set(ylabel='Correlation (r)')
sns.despine(trim=True)
plt.tight_layout()
plt.show()
