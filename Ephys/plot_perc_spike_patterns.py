# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 09:19:20 2025

@author: Guido
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style
colors, dpi = figure_style()

def min_max_scale(x):
    """Calculates Min-Max scaling for a Series x."""
    return (x - x.min()) / (x.max() - x.min())

# Load in data
path_dict = paths()
pattern_n_df = pd.read_csv(join(path_dict['save_path'], 'spike_pattern_n_select.csv'))
pattern_p_df = pd.read_csv(join(path_dict['save_path'], 'spike_pattern_sig.csv'))
regions = np.unique(pattern_n_df['region'])

# Calculate percentage significant per session
pattern_p_df['ses_id'] = pattern_p_df['subject'] + pattern_p_df['date']
pattern_p_df['sig_obj1'] = pattern_p_df['p_obj1'] < 0.05
pattern_p_df['sig_obj2'] = pattern_p_df['p_obj2'] < 0.05
pattern_p_df['sig_sound'] = pattern_p_df['p_sound_id'] < 0.05
pattern_p_df['sig_ripples'] = pattern_p_df['p_ripples'] < 0.05
pattern_p_df['log_p_obj1'] = -np.log10(pattern_p_df['p_obj1'])
pattern_p_df['log_p_obj2'] = -np.log10(pattern_p_df['p_obj2'])
pattern_p_df['log_p_sound'] = -np.log10(pattern_p_df['p_sound_id'])
pattern_p_df['log_p_ripples'] = -np.log10(pattern_p_df['p_ripples'])
per_ses_df = pattern_p_df[['ses_id', 'region']].groupby(['ses_id', 'region']).size().to_frame()
per_ses_df = per_ses_df.rename(columns={0: 'n_patterns'})
per_ses_df['sig_obj1'] = pattern_p_df[['ses_id', 'sig_obj1', 'region']].groupby(['ses_id', 'region']).sum()['sig_obj1']
per_ses_df['sig_obj2'] = pattern_p_df[['ses_id', 'sig_obj2', 'region']].groupby(['ses_id', 'region']).sum()['sig_obj2']
per_ses_df['sig_sound'] = pattern_p_df[['ses_id', 'sig_sound', 'region']].groupby(['ses_id', 'region']).sum()['sig_sound']
per_ses_df['sig_ripples'] = pattern_p_df[['ses_id', 'sig_ripples', 'region']].groupby(['ses_id', 'region']).sum()['sig_ripples']
per_ses_df = per_ses_df.reset_index()
per_ses_df['perc_sig_obj1'] = (per_ses_df['sig_obj1'] / per_ses_df['n_patterns']) * 100
per_ses_df['perc_sig_obj2'] = (per_ses_df['sig_obj2'] / per_ses_df['n_patterns']) * 100
per_ses_df['perc_sig_sound'] = (per_ses_df['sig_sound'] / per_ses_df['n_patterns']) * 100
per_ses_df['perc_sig_ripples'] = (per_ses_df['sig_ripples'] / per_ses_df['n_patterns']) * 100

# %% Do statistics

pattern_p_df = pattern_p_df[~np.isnan(pattern_p_df['p_ripples'])]

# Iterate through regions
f, axs = plt.subplots(1, 6, figsize=(1.75*6, 1.75), dpi=dpi)
for i, (region, group) in enumerate(pattern_p_df.groupby('region')):
        
    # Do Spearman correlation
    #rho, p_val = stats.spearmanr(group['z_obj2'], group['z_ripples'])
    rho, p_val = stats.spearmanr(group['log_p_obj2'], group['log_p_ripples'])
    print(f'{region}; p = {np.round(p_val, 3)}')
    
    # Plot
    #axs[i].scatter(group['z_obj2'], group['z_ripples'], s=3)
    axs[i].scatter(group['log_p_obj2'], group['log_p_ripples'], s=3)
    axs[i].set(title=region, xlabel='Significance object 2 (-log10[p])',
               ylabel='Significance ripples (-log10[p])')

sns.despine(trim=True)
plt.tight_layout()
    

# %%
pattern_n_df['min_max_scaled_log_likelihood'] = (
    pattern_n_df
    .groupby(['region', 'subject', 'date'])
    ['log_likelihood']
    .transform(min_max_scale)
)
f, axs = plt.subplots(2, 3, figsize=(7, 3.5), dpi=dpi, sharey=True, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(regions):
    sns.lineplot(data=pattern_n_df[pattern_n_df['region'] == region],
                 x='n_patterns', y='min_max_scaled_log_likelihood',
                 ax=axs[i], errorbar='se')
    axs[i].set(title=region, xticks=np.arange(1, 13), xlabel='', ylabel='')

# %%

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75*3, 2), dpi=dpi, sharey=True)

this_order = per_ses_df[['region', 'perc_sig_obj1']].groupby('region').mean().sort_values(
    'perc_sig_obj1', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_obj1', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.plot([-0.5, regions.shape[0]-0.5], [5, 5], ls='--', color='grey')
ax1.set(ylabel='Significant patterns (%)',  yticks=[0, 10, 20, 30, 40], xlabel='',
        title='Object 1', ylim=[0, 40], xlim=[-0.5, regions.shape[0]-0.5])
ax1.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_sig_obj2']].groupby('region').mean().sort_values(
    'perc_sig_obj2', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_obj2', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax2.plot([-0.5, regions.shape[0]-0.5], [5, 5], ls='--', color='grey')
ax2.set(ylabel='Significant patterns (%)', xlabel='',
        title='Object 2', xlim=[-0.5, regions.shape[0]-0.5])
ax2.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_sig_sound']].groupby('region').mean().sort_values(
    'perc_sig_sound', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_sound', ax=ax3, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax3.plot([-0.5, regions.shape[0]-0.5], [5, 5], ls='--', color='grey')
ax3.set(ylabel='Significant patterns (%)', xlabel='',
        title='Sound onset', xlim=[-0.5, regions.shape[0]-0.5])
ax3.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'sig_patterns.jpg', dpi=600)

f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
this_order = per_ses_df[['region', 'perc_sig_ripples']].groupby('region').mean().sort_values(
    'perc_sig_ripples', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_ripples', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.plot([-0.5, regions.shape[0]-0.5], [5, 5], ls='--', color='grey')
ax1.set(ylabel='Significant patterns (%)', xlabel='', yticks=np.arange(0, 101, 20),
        title='Ripples', xlim=[-0.5, regions.shape[0]-0.5])
ax1.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'sig_patterns_ripples.jpg', dpi=600)
