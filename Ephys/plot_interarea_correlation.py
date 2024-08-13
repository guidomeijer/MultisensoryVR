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
corr_df = pd.read_csv(join(path_dict['save_path'], 'region_corr_100ms-bins.csv'))

# Plot
region_pairs = np.unique(corr_df['region_pair'])
f, axs = plt.subplots(1, len(region_pairs), figsize=(1.75*len(region_pairs), 1.75), dpi=dpi,
                      sharey=True)

f_sound, axs_sound = plt.subplots(
    1, len(region_pairs), figsize=(1.75*len(region_pairs), 1.75), dpi=dpi,
    sharey=True)
for i, region_pair in enumerate(region_pairs):
    
    """
    long_df = pd.melt(corr_df[corr_df['region_pair'] == region_pair],
                      id_vars='time', value_vars=['r_goal_baseline', 'r_distractor_baseline'])
    sns.lineplot(data=long_df, x='time', y='value', hue='variable', ax=axs[i], errorbar='se',
                 legend=None, hue_order=['r_goal_baseline', 'r_distractor_baseline'],
                 palette=[colors['goal'], colors['no-goal']])
    """
    this_df = corr_df[corr_df['region_pair'] == region_pair]
    axs[i].plot(this_df['time'], this_df['r_goal_baseline'], color=colors['goal'])
    axs[i].fill_between(this_df['time'],
                        this_df['r_goal_baseline'] - this_df['r_sem_goal'],
                        this_df['r_goal_baseline'] + this_df['r_sem_goal'],
                        color=colors['goal'], alpha=0.25, lw=0)
    axs[i].plot(this_df['time'], this_df['r_distractor_baseline'], color=colors['no-goal'])
    axs[i].fill_between(this_df['time'],
                        this_df['r_distractor_baseline'] - this_df['r_sem_distractor'],
                        this_df['r_distractor_baseline'] + this_df['r_sem_distractor'],
                        color=colors['no-goal'], alpha=0.25, lw=0)
    
    axs[i].set(title=f'{region_pair}')
    
    axs_sound[i].plot(this_df['time'], this_df['r_sound_baseline'], color=colors['goal'])
    axs_sound[i].fill_between(this_df['time'],
                        this_df['r_sound_baseline'] - this_df['r_sem_sound'],
                        this_df['r_sound_baseline'] + this_df['r_sem_sound'],
                        color=colors['goal'], alpha=0.25, lw=0)
    axs_sound[i].plot(axs_sound[i].get_xlim(), [0, 0], color='grey', ls='--')
    
    axs_sound[i].set(title=f'{region_pair}')
    
axs[0].set(ylabel='Correlation (r)', yticks=[-0.04, 0, 0.04])
sns.despine(trim=True)
plt.tight_layout()
