# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:29:49 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style, load_subjects
colors, dpi = figure_style()

# Load in data
path_dict = paths()
subjects = load_subjects()
stats_df = pd.read_csv(join(path_dict['save_path'], 'ramping_neurons.csv'))
#stats_df = stats_df[np.isin(stats_df['subject'],
#                            subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
stats_df['subject'] = stats_df['subject'].astype(str)
stats_df['date'] = stats_df['date'].astype(str)

# Take out iCA1 for now
stats_df.loc[stats_df['region'] == 'dCA1', 'region'] = 'CA1'
stats_df.loc[stats_df['region'] == 'iCA1', 'region'] = 'CA1'
#stats_df = stats_df[stats_df['region'] != 'iCA1']
session_df = stats_df[['subject', 'date', 'probe']].value_counts().reset_index()

# Do some processing
stats_df['sig_context'] = stats_df['p_context'] < 0.05
stats_df['sig_ramp'] = stats_df['p_ramp'] < 0.05
stats_df['sig_both'] = stats_df['sig_context'] & stats_df['sig_ramp']
stats_df = stats_df[stats_df['region'] != 'root']
stats_df['ses_id'] = [f'{stats_df.loc[i, "subject"]}_{stats_df.loc[i, "date"]}' for i in stats_df.index]

# Summary statistics per session
per_ses_df = stats_df.groupby(['region', 'ses_id']).sum(numeric_only=True)
per_ses_df['n_neurons'] = stats_df.groupby(['region', 'ses_id']).size()
per_ses_df['n_sig_neurons'] = stats_df[stats_df['sig_context']].groupby(['region', 'ses_id']).size()
per_ses_df['perc_ramp'] = (per_ses_df['sig_ramp'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_both'] = (per_ses_df['sig_both'] / per_ses_df['n_sig_neurons']) * 100
per_ses_df = per_ses_df.reset_index()

# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.6*2, 2), dpi=dpi, sharey=False)

this_order = per_ses_df[['region', 'perc_ramp']].groupby('region').mean().sort_values(
    'perc_ramp', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_ramp', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
#ax1.set(ylabel='Significant neurons (%)',  yticks=[0, 20, 40, 60, 80], xlabel='',
#        title='Object modulation', ylim=[0, 80])
ax1.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_both']].groupby('region').mean().sort_values(
    'perc_both', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_both', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=this_order)
#ax2.set(xlabel='', title='Context first landmark', yticks=[0, 1, 2, 3, 4, 5, 6], ylim=[0, 6], ylabel='')
#ax2.set(xlabel='', title='Context modulation (first object)', yticks=[0, 5, 10, 15], ylim=[0, 15], ylabel='')
ax2.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.show(block=False)

plt.savefig(join(path_dict['google_drive_fig_path'], 'perc_ramp_neurons.jpg'), dpi=600)
#plt.savefig(join(path_dict['google_drive_fig_path'], 'perc_ramp_neurons.pdf'))


