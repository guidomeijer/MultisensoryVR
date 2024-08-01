# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:44:26 2024 by Guido Meijer
"""


import numpy as np
import os
from os.path import join, isdir
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from msvr_functions import paths, figure_style, combine_regions

# Settings
MIN_N_BINS = 30
ALPHA = 0.05
colors, dpi = figure_style()

# Load in data
path_dict = paths()
glm_df = pd.read_csv(join(path_dict['save_path'], 'glm_results.csv'))
glm_df['region'] = combine_regions(glm_df['allen_acronym'])
glm_df = glm_df[glm_df['region'] != 'root']
glm_df['coef_object_abs'] = np.abs(glm_df['coef_object'])
glm_df['coef_sound_abs'] = np.abs(glm_df['coef_sound'])
glm_df['coef_goal_abs'] = np.abs(glm_df['coef_goal'])
glm_df = glm_df.sort_values(by=['neuron_id', 'time']).reset_index(drop=True)

# Loop over neurons
stats_df = pd.DataFrame()
for i, neuron_id in enumerate(np.unique(glm_df['neuron_id'])):
    
    this_df = glm_df[glm_df['neuron_id'] == neuron_id]
    if np.sum(np.isnan(this_df['p_object'])) == this_df.shape[0]:
        continue
    
    p_object = this_df.loc[~np.isnan(this_df['p_object']), 'p_object'].values
    _, p_object_corr, _, _ = multipletests(p_object, alpha=0.05, method='fdr_bh')
    p_sound = this_df.loc[~np.isnan(this_df['p_sound']), 'p_sound'].values
    _, p_sound_corr, _, _ = multipletests(p_sound, alpha=0.05, method='fdr_bh')
    p_goal = this_df.loc[~np.isnan(this_df['p_goal']), 'p_goal'].values
    _, p_goal_corr, _, _ = multipletests(p_goal, alpha=0.05, method='fdr_bh')
    stats_df = pd.concat((stats_df, pd.DataFrame(index=[stats_df.shape[0]], data={
        'neuron_id': neuron_id,
        'region': this_df['region'].values[0],
        'p_object': np.min(p_object_corr),
        'n_object': np.sum(p_object_corr < ALPHA),
        'p_sound': np.min(p_sound_corr),
        'n_sound': np.sum(p_sound_corr < ALPHA),
        'p_goal': np.min(p_goal_corr),
        'n_goal': np.sum(p_goal_corr < ALPHA)
        })))
    
    
# %%
# Determine significant neurons
stats_df['sig_object'] = (stats_df['n_object'] >= MIN_N_BINS).astype(int)
stats_df['sig_sound'] = (stats_df['n_sound'] >= MIN_N_BINS).astype(int)
stats_df['sig_goal'] = (stats_df['n_goal'] >= MIN_N_BINS).astype(int)
  
# Get percentage of significant neurons per region  
region_df = stats_df.groupby('region').sum()
region_df['n_neurons'] = stats_df.groupby('region').size()
region_df['perc_object'] = (region_df['sig_object'] / region_df['n_neurons']) * 100
region_df['perc_goal'] = (region_df['sig_goal'] / region_df['n_neurons']) * 100
region_df['perc_sound'] = (region_df['sig_sound'] / region_df['n_neurons']) * 100

# Add to dataframe
glm_df['sig_object'] = np.isin(glm_df['neuron_id'], stats_df.loc[stats_df['sig_object'] == 1, 'neuron_id']).astype(int)
glm_df['sig_sound'] = np.isin(glm_df['neuron_id'], stats_df.loc[stats_df['sig_sound'] == 1, 'neuron_id']).astype(int)
glm_df['sig_goal'] = np.isin(glm_df['neuron_id'], stats_df.loc[stats_df['sig_goal'] == 1, 'neuron_id']).astype(int)

# %% Plot coefficients per region
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 1.75), dpi=dpi)

sns.lineplot(data=glm_df[glm_df['sig_object'] == 1], x='time', y='coef_object_abs', ax=ax1,
             errorbar='se', hue='region')

sns.lineplot(data=glm_df[glm_df['sig_sound'] == 1], x='time', y='coef_sound_abs', ax=ax2,
             errorbar='se', hue='region')

sns.lineplot(data=glm_df[glm_df['sig_goal'] == 1], x='time', y='coef_goal_abs', ax=ax3,
             errorbar='se', hue='region')



