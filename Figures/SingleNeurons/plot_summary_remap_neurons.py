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
stats_df = pd.read_csv(join(path_dict['save_path'], 'remapping_neurons.csv'),
                       dtype={'subject': str, 'date': str})
session_df = stats_df[['subject', 'date', 'probe']].value_counts().reset_index()

print(f'{len(np.unique(session_df["subject"]))} mice')
print(f'{len(np.unique(session_df["date"]))} recording sessions')
print(f'{session_df.shape[0]} probe insertions')
print(f'{stats_df.shape[0]} neurons ({int(session_df["count"].mean())} +- {int(session_df["count"].sem())}, mean +- sem per probe)')

# Do some processing
stats_df['sig_obj1'] = stats_df['p_obj1'] < 0.05
stats_df['sig_obj2'] = stats_df['p_obj2'] < 0.05
stats_df = stats_df[stats_df['region'] != 'root']
stats_df['ses_id'] = [f'{stats_df.loc[i, "subject"]}_{stats_df.loc[i, "date"]}' for i in stats_df.index]

# Summary statistics per session
per_ses_df = stats_df.groupby(['region', 'ses_id']).sum(numeric_only=True)
per_ses_df['n_neurons'] = stats_df.groupby(['region', 'ses_id']).size()
per_ses_df['perc_obj1'] = (per_ses_df['sig_obj1'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_obj2'] = (per_ses_df['sig_obj2'] / per_ses_df['n_neurons']) * 100
per_ses_df = per_ses_df.reset_index()

# Plot number of neurons per region
region_summary = per_ses_df.groupby('region')['n_neurons'].agg(['mean', 'sum', 'sem'])
region_summary = region_summary.sort_values(by='sum', ascending=False)
print(region_summary)


# %%
use_xlim = [-0.7, 5.5]
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.5*2, 2), dpi=dpi, sharey=False)

this_order = per_ses_df[['region', 'perc_obj1']].groupby('region').mean().sort_values(
    'perc_obj1', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_obj1', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)

ax1.set(ylabel='Significant neurons (%)',  yticks=[0, 10, 20, 30], xlabel='',
        title='Object 1', ylim=[0, 30], xlim=use_xlim)
ax1.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_obj2']].groupby('region').mean().sort_values(
    'perc_obj2', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_obj2', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=this_order)

#ax2.set(xlabel='', title='Context first landmark', yticks=[0, 1, 2, 3, 4, 5, 6], ylim=[0, 6], ylabel='')
ax2.set(xlabel='', title='Object 2', yticks=[0, 10, 20, 30], ylim=[0, 30], ylabel='', xlim=use_xlim)
ax2.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'SingleNeurons' / 'perc_remap_neurons.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'SingleNeurons' / 'perc_remap_neurons.pdf')
plt.show(block=False)

