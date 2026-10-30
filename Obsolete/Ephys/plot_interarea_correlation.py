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


# %% Plot all

for region in np.unique(corr_df['region_1']):
    
    f, axs = plt.subplots(1, 3, figsize=(1.75*3, 1.75), dpi=dpi)
    
    sns.lineplot(corr_df[corr_df['region_1'] == region], x='time', y='r_goal_baseline', hue='region_pair',
                 ax=axs[0], errorbar='se', err_kws={'lw': 0}, legend=None)
    
    sns.lineplot(corr_df[corr_df['region_1'] == region], x='time', y='r_distractor_baseline', hue='region_pair',
                 ax=axs[1], errorbar='se', err_kws={'lw': 0}, legend=None)
    
    sns.lineplot(corr_df[corr_df['region_1'] == region], x='time', y='r_sound_baseline', hue='region_pair',
                 ax=axs[2], errorbar='se', err_kws={'lw': 0})
    
    sns.despine(trim=True)
    plt.tight_layout()


# %% Plot individual examples

f, axs = plt.subplots(1, 3, figsize=(5.25, 2), dpi=dpi, sharey=True)

this_df = pd.melt(corr_df[corr_df['region_pair'] == 'PERI 35-dCA1'], id_vars=['time'],
                  value_vars=['r_goal_baseline', 'r_distractor_baseline'])
sns.lineplot(this_df, x='time', y='value', hue='variable', ax=axs[0], errorbar='se')

this_df = pd.melt(corr_df[corr_df['region_pair'] == 'PERI 36-TEa'], id_vars=['time'],
                  value_vars=['r_goal_baseline', 'r_distractor_baseline'])
sns.lineplot(this_df, x='time', y='value', hue='variable', ax=axs[1], errorbar='se')

this_df = pd.melt(corr_df[corr_df['region_pair'] == 'PERI 35-PERI 36'], id_vars=['time'],
                  value_vars=['r_goal_baseline', 'r_distractor_baseline'])
sns.lineplot(this_df, x='time', y='value', hue='variable', ax=axs[2], errorbar='se')


sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(path_dict['google_drive_fig_path'], 'interarea_correlation_goal_obj_entry.jpg'), dpi=600)

# %%
f, axs = plt.subplots(1, 3, figsize=(5.25, 2), dpi=dpi, sharey=True)

sns.lineplot(corr_df[corr_df['region_pair'] == 'PERI 35-dCA1'], x='time', y='r_sound_baseline',
             ax=axs[0], errorbar='se')

sns.lineplot(corr_df[corr_df['region_pair'] == 'PERI 36-TEa'], x='time', y='r_sound_baseline',
             ax=axs[1], errorbar='se')

sns.lineplot(corr_df[corr_df['region_pair'] == 'PERI 35-PERI 36'], x='time', y='r_sound_baseline',
             ax=axs[2], errorbar='se')


sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(path_dict['google_drive_fig_path'], 'interarea_correlation_sound.jpg'), dpi=600)

