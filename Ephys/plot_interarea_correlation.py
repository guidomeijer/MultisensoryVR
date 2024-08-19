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

# %% Plot individual examples

f, axs = plt.subplots(1, 3, figsize=(5.25, 2), dpi=dpi, sharey=True)

this_df = corr_df[corr_df['region_pair'] == 'sPERI-HPC']
axs[0].plot(this_df['time'], this_df['r_goal_baseline'], color=colors['goal'], label='Goal')
axs[0].fill_between(this_df['time'],
                    this_df['r_goal_baseline'] - this_df['r_sem_goal'],
                    this_df['r_goal_baseline'] + this_df['r_sem_goal'],
                    color=colors['goal'], alpha=0.25, lw=0)
axs[0].plot(this_df['time'], this_df['r_distractor_baseline'], color=colors['no-goal'], label='No goal')
axs[0].fill_between(this_df['time'],
                    this_df['r_distractor_baseline'] - this_df['r_sem_distractor'],
                    this_df['r_distractor_baseline'] + this_df['r_sem_distractor'],
                    color=colors['no-goal'], alpha=0.25, lw=0)
axs[0].set(xlabel='Time from object entry (s)', ylabel='Pairwise correlation (r)',
           title='sPERI - HPC', ylim=[-0.02, 0.03], yticks=[-0.02, 0, 0.03])
axs[0].legend(prop={'size': 5})

this_df = corr_df[corr_df['region_pair'] == 'dPERI-HPC']
axs[1].plot(this_df['time'], this_df['r_goal_baseline'], color=colors['goal'])
axs[1].fill_between(this_df['time'],
                    this_df['r_goal_baseline'] - this_df['r_sem_goal'],
                    this_df['r_goal_baseline'] + this_df['r_sem_goal'],
                    color=colors['goal'], alpha=0.25, lw=0)
axs[1].plot(this_df['time'], this_df['r_distractor_baseline'], color=colors['no-goal'])
axs[1].fill_between(this_df['time'],
                    this_df['r_distractor_baseline'] - this_df['r_sem_distractor'],
                    this_df['r_distractor_baseline'] + this_df['r_sem_distractor'],
                    color=colors['no-goal'], alpha=0.25, lw=0)
axs[1].set(xlabel='Time from object entry (s)', title='dPERI - HPC')

this_df = corr_df[corr_df['region_pair'] == 'TEa-HPC']
axs[2].plot(this_df['time'], this_df['r_goal_baseline'], color=colors['goal'])
axs[2].fill_between(this_df['time'],
                    this_df['r_goal_baseline'] - this_df['r_sem_goal'],
                    this_df['r_goal_baseline'] + this_df['r_sem_goal'],
                    color=colors['goal'], alpha=0.25, lw=0)
axs[2].plot(this_df['time'], this_df['r_distractor_baseline'], color=colors['no-goal'])
axs[2].fill_between(this_df['time'],
                    this_df['r_distractor_baseline'] - this_df['r_sem_distractor'],
                    this_df['r_distractor_baseline'] + this_df['r_sem_distractor'],
                    color=colors['no-goal'], alpha=0.25, lw=0)
axs[2].set(xlabel='Time from object entry (s)', title='TEa - HPC')

sns.despine(trim=True)
plt.tight_layout()

# %% Plot individual examples
plot_pairs = ['TEa-sPERI', 'sPERI-HPC', 'TEa-HPC']
these_colors = sns.color_palette('Set2')[:3]

f, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi, sharey=True)

for i, region_pair in enumerate(plot_pairs):
    this_df = corr_df[corr_df['region_pair'] == region_pair]
    ax.plot(this_df['time'], this_df['r_sound_baseline'], color=these_colors[i], label=region_pair)
    ax.fill_between(this_df['time'],
                        this_df['r_sound_baseline'] - this_df['r_sem_sound'],
                        this_df['r_sound_baseline'] + this_df['r_sem_sound'],
                        color=these_colors[i], alpha=0.25, lw=0)
   
ax.set(xlabel='Time from sound onset (s)', ylabel='Pairwise correlation (r)',
       ylim=[-0.04, 0.04], yticks=[-0.04, 0, 0.04])
ax.legend(prop={'size': 5})



sns.despine(trim=True)
plt.tight_layout()
