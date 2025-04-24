# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:29:49 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style, combine_regions
colors, dpi = figure_style()

# Load in data
path_dict = paths()
stats_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
session_df = stats_df[['subject', 'date', 'probe']].value_counts().reset_index()
print(f'{len(np.unique(session_df["subject"]))} mice')
print(f'{len(np.unique(session_df["date"]))} recording sessions')
print(f'{session_df.shape[0]} probe insertions')
print(f'{stats_df.shape[0]} neurons ({int(session_df["count"].mean())} +- {int(session_df["count"].sem())}, mean +- sem per probe)')

# Select neurons
stats_df = stats_df[stats_df['region'] != 'root']
stats_df = stats_df[stats_df['region'] != 'ENT']
stats_df['ses_id'] = [f'{stats_df.loc[i, "subject"]}_{stats_df.loc[i, "date"]}' for i in stats_df.index]

# Summary statistics per session
per_ses_df = stats_df.groupby(['region', 'ses_id']).sum(numeric_only=True)
per_ses_df['n_neurons'] = stats_df.groupby(['region', 'ses_id']).size()
per_ses_df['perc_context_obj1'] = (per_ses_df['sig_context_obj1'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_context_obj2'] = (per_ses_df['sig_context_obj2'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_context_onset'] = (per_ses_df['sig_sound_onset'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_reward'] = (per_ses_df['sig_reward'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_omission'] = (per_ses_df['sig_omission'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_obj_onset'] = (per_ses_df['sig_obj_onset'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_control'] = (per_ses_df['sig_control'] / per_ses_df['n_neurons']) * 100
per_ses_df = per_ses_df.reset_index()
all_nan_omission = stats_df.groupby(['region', 'ses_id'])['p_omission'].apply(lambda x: x.isna().all())
per_ses_df.loc[per_ses_df.set_index(['region', 'ses_id']).index.isin(all_nan_omission[all_nan_omission].index), 'perc_omission'] = np.nan


# %%
f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(1.75*4, 4), dpi=dpi, sharey=False) 

sns.barplot(data=per_ses_df, x='region', y='perc_obj_onset', ax=ax1, hue='region', errorbar=None,
            palette=colors)
sns.swarmplot(data=per_ses_df, x='region', y='perc_obj_onset', ax=ax1, color='k', size=2, clip_on=False)
ax1.set(ylabel='Significant neurons (%)',  yticks=[0, 20, 40, 60, 80, 100], xlabel='',
        title='Landmark entries', ylim=[0, 100])
ax1.tick_params(axis='x', labelrotation=90)

sns.barplot(data=per_ses_df, x='region', y='perc_context_obj1', ax=ax2, hue='region', errorbar=None,
            palette=colors)
sns.swarmplot(data=per_ses_df, x='region', y='perc_context_obj1', ax=ax2, color='k', size=2)
ax2.set(xlabel='', title='Context first landmark', yticks=[0, 5, 10, 15, 20, 25, 30, 35], ylim=[0, 35], ylabel='')
ax2.tick_params(axis='x', labelrotation=90)

sns.barplot(data=per_ses_df, x='region', y='perc_context_obj2', ax=ax3, hue='region', errorbar=None,
            palette=colors)
sns.swarmplot(data=per_ses_df, x='region', y='perc_context_obj2', ax=ax3, color='k', size=2)
ax3.set(xlabel='', title='Context second landmark', yticks=[0, 5, 10, 15, 20, 25, 30, 35], ylim=[0, 35], ylabel='')
ax3.tick_params(axis='x', labelrotation=90)

sns.barplot(data=per_ses_df, x='region', y='perc_context_onset', ax=ax4, hue='region', errorbar=None,
            palette=colors)
sns.swarmplot(data=per_ses_df, x='region', y='perc_context_onset', ax=ax4, color='k', size=2)
ax4.set(xlabel='', title='Context onset', yticks=[0, 10, 20, 30, 40, 50, 60, 70], ylim=[0, 70], ylabel='')
ax4.tick_params(axis='x', labelrotation=90)

sns.barplot(data=per_ses_df, x='region', y='perc_reward', ax=ax5, hue='region', errorbar=None,
            palette=colors)
sns.swarmplot(data=per_ses_df, x='region', y='perc_reward', ax=ax5, color='k', size=2)
ax5.set(xlabel='', title='Outcome', yticks=[0, 10, 20, 30, 40, 50], ylim=[0, 50], ylabel='')
ax5.tick_params(axis='x', labelrotation=90)

sns.barplot(data=per_ses_df, x='region', y='perc_omission', ax=ax6, hue='region', errorbar=None,
            palette=colors)
sns.swarmplot(data=per_ses_df, x='region', y='perc_omission', ax=ax6, color='k', size=2)
ax6.set(xlabel='', title='Reward omission', yticks=[0, 10, 20, 30, 40, 50, 60], ylim=[0, 60], ylabel='')
ax6.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.show(block=False)

plt.savefig(join(path_dict['google_drive_fig_path'], 'perc_sig_neurons_swarm.jpg'), dpi=600)

# %%
f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(1.75*4, 4), dpi=dpi, sharey=False) 

this_order = per_ses_df[['region', 'perc_obj_onset']].groupby('region').mean().sort_values(
    'perc_obj_onset', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_obj_onset', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.set(ylabel='Significant neurons (%)',  yticks=[0, 20, 40, 60, 80, 100], xlabel='',
        title='Landmark entries', ylim=[0, 100])
ax1.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_context_obj1']].groupby('region').mean().sort_values(
    'perc_context_obj1', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_context_obj1', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax2.set(xlabel='', title='Context first landmark', yticks=[0, 1, 2, 3, 4, 5, 6], ylim=[0, 6], ylabel='')
ax2.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_context_obj2']].groupby('region').mean().sort_values(
    'perc_context_obj2', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_context_obj2', ax=ax3, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax3.set(xlabel='', title='Context second landmark', yticks=[0, 5, 10, 15, 20, 25, 30], ylim=[0, 30], ylabel='')
ax3.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_context_onset']].groupby('region').mean().sort_values(
    'perc_context_onset', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_context_onset', ax=ax4, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax4.set(xlabel='', title='Context onset', yticks=[0, 10, 20, 30, 40, 50], ylim=[0, 50], ylabel='')
ax4.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_reward']].groupby('region').mean().sort_values(
    'perc_reward', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_reward', ax=ax5, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax5.set(xlabel='', title='Outcome', yticks=[0, 10, 20, 30, 40], ylim=[0, 40], ylabel='')
ax5.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_omission']].groupby('region').mean().sort_values(
    'perc_omission', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_omission', ax=ax6, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax6.set(xlabel='', title='Reward omission', yticks=[0, 10, 20, 30, 40], ylim=[0, 40], ylabel='')
ax6.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.show(block=False)

plt.savefig(join(path_dict['google_drive_fig_path'], 'perc_sig_neurons_errorbars.jpg'), dpi=600)


# %%

sig_cols = [
    'sig_context_obj1',
    'sig_context_obj2',
    'sig_reward',
    'sig_omission',
    'sig_obj_onset'
]

# Initialize an empty DataFrame to store percentage overlaps
overlap_matrix = pd.DataFrame(index=sig_cols, columns=sig_cols, dtype=float)

for col1 in sig_cols:
    for col2 in sig_cols:
        intersection = (stats_df[col1] & stats_df[col2]).sum()
        union = (stats_df[col1] | stats_df[col2]).sum()
        overlap_percent = 100 * intersection / union if union > 0 else 0
        overlap_matrix.loc[col1, col2] = overlap_percent
        
# Assume overlap_matrix is your symmetric matrix of floats
# Get lower triangle indices (excluding diagonal if desired)
lower_triangle_indices = np.tril_indices_from(overlap_matrix, k=-1)  # use k=0 to include diagonal

# Find the rows and columns that actually have values
rows_to_keep = sorted(set(lower_triangle_indices[0]))
cols_to_keep = sorted(set(lower_triangle_indices[1]))

# Subset the matrix
trimmed_matrix = overlap_matrix.iloc[rows_to_keep, cols_to_keep]

# Create new mask for upper triangle (on the trimmed matrix)
mask = np.triu(np.ones_like(trimmed_matrix, dtype=bool), k=1)

# Plot
plt.figure(figsize=(3.5, 3), dpi=dpi)
ax = sns.heatmap(
        trimmed_matrix.astype(float),
        annot=True,
        fmt=".1f",
        cmap="viridis",
        vmin=0,
        vmax=30,
        mask=mask,
        cbar=None,
        linewidths=0.5,
        linecolor='white'
        )
ax.set_xticklabels(['Object 1', 'Object 2', 'Outcome', 'Omission'], rotation=0)
ax.set_yticklabels(['Object 2', 'Outcome', 'Omission', 'Landmark entry'], rotation=0)
plt.tight_layout()

plt.savefig(join(path_dict['google_drive_fig_path'], 'perc_sig_neurons_overlap.jpg'), dpi=600)
