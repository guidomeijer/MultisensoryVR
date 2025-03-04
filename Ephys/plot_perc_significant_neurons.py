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
stats_df['region'] = combine_regions(stats_df['allen_acronym'], split_peri=True, abbreviate=True)
stats_df = stats_df[stats_df['region'] != 'root']
stats_df = stats_df[stats_df['region'] != 'ENT']
stats_df['sig_goal_no_control'] = stats_df['sig_goal'] & ~stats_df['sig_control']
stats_df['ses_id'] = [f'{stats_df.loc[i, "subject"]}_{stats_df.loc[i, "date"]}' for i in stats_df.index]

# Summary statistics per session
per_ses_df = stats_df.groupby(['region', 'ses_id']).sum(numeric_only=True)
per_ses_df['n_neurons'] = stats_df.groupby(['region', 'ses_id']).size()
per_ses_df['perc_goal'] = (per_ses_df['sig_goal'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_obj_onset'] = (per_ses_df['sig_obj_onset'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_control'] = (per_ses_df['sig_control'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_obj_diff'] = (per_ses_df['sig_obj_diff'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_goal_no_control'] = (per_ses_df['sig_goal_no_control'] / per_ses_df['n_neurons']) * 100
per_ses_df = per_ses_df.reset_index()

# Get percentage of significant neurons per region
region_df = stats_df[['region', 'sig_goal', 'sig_obj_onset', 'sig_control', 'sig_obj_diff',
                      'sig_goal_no_control']].groupby('region').sum()
region_df['n_neurons'] = stats_df.groupby('region').size()
region_df['perc_goal'] = (region_df['sig_goal'] / region_df['n_neurons']) * 100
region_df['perc_obj_onset'] = (region_df['sig_obj_onset'] / region_df['n_neurons']) * 100
region_df['perc_control'] = (region_df['sig_control'] / region_df['n_neurons']) * 100
region_df['perc_obj_diff'] = (region_df['sig_obj_diff'] / region_df['n_neurons']) * 100
region_df['perc_goal_no_control'] = (region_df['sig_goal_no_control'] / region_df['n_neurons']) * 100
region_df = region_df.reset_index()

long_df = pd.melt(region_df, id_vars=['region'],
                  value_vars=['perc_obj_onset', 'perc_goal', 'perc_goal_no_control', 'perc_obj_diff'])



# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.75*2, 2), dpi=dpi) 

sns.barplot(data=per_ses_df, x='region', y='perc_obj_onset', ax=ax1, hue='region', errorbar=None,
            palette=colors)
sns.swarmplot(data=per_ses_df, x='region', y='perc_obj_onset', ax=ax1, color='k', size=3)
ax1.set(ylabel='Significant neurons (%)',  yticks=[0, 20, 40, 60, 80, 100], xlabel='',
        title='Landmark')
ax1.tick_params(axis='x', labelrotation=90)

sns.barplot(data=per_ses_df, x='region', y='perc_goal', ax=ax2, hue='region', errorbar=None,
            palette=colors)
sns.swarmplot(data=per_ses_df, x='region', y='perc_goal', ax=ax2, color='k', size=3)
ax2.set(xlabel='', title='Context', yticks=[0, 10, 20, 30, 40, 50], ylabel='Significant neurons (%)')
ax2.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()

plt.savefig(join(path_dict['google_drive_fig_path'], 'perc_sig_neurons_per_ses.jpg'), dpi=600)

