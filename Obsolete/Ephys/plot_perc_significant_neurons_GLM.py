# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:29:49 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style
colors, dpi = figure_style()

# Load in data
path_dict = paths()
stats_df = pd.read_csv(join(path_dict['save_path'], 'GLM_results.csv'))
stats_df['subject'] = stats_df['subject'].astype(str)
stats_df['date'] = stats_df['date'].astype(str)

# Take out iCA1 for now
stats_df.loc[stats_df['region'] == 'dCA1', 'region'] = 'CA1'
stats_df = stats_df[stats_df['region'] != 'iCA1']
session_df = stats_df[['subject', 'date', 'probe']].value_counts().reset_index()

# Do some processing
stats_df['sig_object1'] = stats_df['no_reward_at_object2'] < 0.05
stats_df['sig_expect'] = stats_df['expect_state'] < 0.05
stats_df['sig_position'] = stats_df['position'] < 0.05
stats_df['sig_acceleration'] = stats_df['acceleration'] < 0.05
stats_df['sig_speed'] = stats_df['speed'] < 0.05
stats_df['sig_lick'] = stats_df['lick'] < 0.05
stats_df = stats_df[stats_df['region'] != 'root']
stats_df['ses_id'] = [f'{stats_df.loc[i, "subject"]}_{stats_df.loc[i, "date"]}' for i in stats_df.index]

# Summary statistics per session
per_ses_df = stats_df.groupby(['region', 'ses_id']).sum(numeric_only=True)
per_ses_df['n_neurons'] = stats_df.groupby(['region', 'ses_id']).size()
per_ses_df['perc_object1'] = (per_ses_df['sig_object1'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_expect'] = (per_ses_df['sig_expect'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_speed'] = (per_ses_df['sig_speed'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_acceleration'] = (per_ses_df['sig_acceleration'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_position'] = (per_ses_df['sig_position'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_lick'] = (per_ses_df['sig_lick'] / per_ses_df['n_neurons']) * 100
per_ses_df = per_ses_df.reset_index()

# %%
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(1.6*5, 2), dpi=dpi, sharey=False)

this_order = per_ses_df[['region', 'perc_position']].groupby('region').mean().sort_values(
    'perc_position', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_position', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.set(ylabel='Significant neurons (%)', xlabel='', title='Position',
        yticks=[0, 20, 40, 60, 80, 100], ylim=[0, 100])
ax1.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_speed']].groupby('region').mean().sort_values(
    'perc_speed', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_speed', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax2.set(yticks=[0, 20, 40, 60, 80, 100], xlabel='', ylabel='',
        title='Speed', ylim=[0, 100])
ax2.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_expect']].groupby('region').mean().sort_values(
    'perc_expect', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_expect', ax=ax3, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax3.set(xlabel='', title='Reward expectancy', yticks=[0, 20, 40, 60, 80, 100], ylim=[0, 100], ylabel='')
ax3.tick_params(axis='x', labelrotation=90)



sns.despine(trim=False)
plt.tight_layout()
plt.show(block=False)

plt.savefig(join(path_dict['google_drive_fig_path'], 'perc_sig_neurons_GLM.jpg'), dpi=600)


