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
stats_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
stats_df['subject'] = stats_df['subject'].astype(str)
stats_df['date'] = stats_df['date'].astype(str)
session_df = stats_df[['subject', 'date', 'probe']].value_counts().reset_index()

print(f'{len(np.unique(session_df["subject"]))} mice')
print(f'{len(np.unique(session_df["date"]))} recording sessions')
print(f'{session_df.shape[0]} probe insertions')
print(f'{stats_df.shape[0]} neurons ({int(session_df["count"].mean())} +- {int(session_df["count"].sem())}, mean +- sem per probe)')

# Do some processing
stats_df['sig_context_obj1'] = stats_df['p_context_obj1'] < 0.05
stats_df['sig_context_obj2'] = stats_df['p_context_obj2'] < 0.05
stats_df['sig_context_onset'] = stats_df['p_sound_onset'] < 0.05
stats_df['sig_context_diff'] = stats_df['p_sound_diff'] < 0.05
stats_df['sig_reward'] = stats_df['p_reward'] < 0.05
stats_df['sig_obj_onset'] = stats_df['p_obj_onset'] < 0.05
stats_df = stats_df[stats_df['region'] != 'root']
stats_df['ses_id'] = [f'{stats_df.loc[i, "subject"]}_{stats_df.loc[i, "date"]}' for i in stats_df.index]

# Get chance levels
context_obj1_chance = (np.sum(stats_df['p_context_obj1_shuf'] < 0.05) / stats_df.shape[0]) * 100
context_obj2_chance = (np.sum(stats_df['p_context_obj2_shuf'] < 0.05) / stats_df.shape[0]) * 100
context_onset_chance = (np.sum(stats_df['p_sound_onset_shuf'] < 0.05) / stats_df.shape[0]) * 100
reward_chance = (np.sum(stats_df['p_reward_onset_shuf'] < 0.05) / stats_df.shape[0]) * 100
obj_onset_chance = (np.sum(stats_df['p_obj_onset_shuf'] < 0.05) / stats_df.shape[0]) * 100

# Summary statistics per session
per_ses_df = stats_df.groupby(['region', 'ses_id']).sum(numeric_only=True)
per_ses_df['n_neurons'] = stats_df.groupby(['region', 'ses_id']).size()
per_ses_df['perc_context_obj1'] = (per_ses_df['sig_context_obj1'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_context_obj2'] = (per_ses_df['sig_context_obj2'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_context_onset'] = (per_ses_df['sig_context_onset'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_context_diff'] = (per_ses_df['sig_context_diff'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_reward'] = (per_ses_df['sig_reward'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_obj_onset'] = (per_ses_df['sig_obj_onset'] / per_ses_df['n_neurons']) * 100
per_ses_df = per_ses_df.reset_index()

# Plot number of neurons per region
merge_peri = per_ses_df.copy()
merge_peri.loc[merge_peri['region'] == 'PERI 36', 'region'] = 'PERI'
merge_peri.loc[merge_peri['region'] == 'PERI 35', 'region'] = 'PERI'
over_ses = merge_peri[['ses_id', 'region', 'n_neurons']].groupby(['ses_id', 'region', 'n_neurons']).sum().reset_index()
region_summary = over_ses.groupby('region')['n_neurons'].agg(['mean', 'sum', 'sem'])
region_summary = region_summary.sort_values(by='sum', ascending=False)
print(region_summary)


# %%
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(1.6*5, 2), dpi=dpi, sharey=False)

this_order = per_ses_df[['region', 'perc_obj_onset']].groupby('region').mean().sort_values(
    'perc_obj_onset', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_obj_onset', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.plot(ax1.get_xlim(), [obj_onset_chance, obj_onset_chance], ls='--', color='lightgrey',
         lw=0.75)
ax1.set(ylabel='Significant neurons (%)',  yticks=[0, 20, 40, 60, 80], xlabel='',
        title='Object modulation', ylim=[0, 80])
ax1.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_context_obj1']].groupby('region').mean().sort_values(
    'perc_context_obj1', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_context_obj1', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax2.plot(ax1.get_xlim(), [context_obj1_chance, context_obj1_chance], ls='--', color='lightgrey',
         lw=0.75)
#ax2.set(xlabel='', title='Context first landmark', yticks=[0, 1, 2, 3, 4, 5, 6], ylim=[0, 6], ylabel='')
ax2.set(xlabel='', title='Context modulation (first object)', yticks=[0, 5, 10, 15], ylim=[0, 15], ylabel='')
ax2.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_context_obj2']].groupby('region').mean().sort_values(
    'perc_context_obj2', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_context_obj2', ax=ax3, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax3.plot(ax1.get_xlim(), [context_obj2_chance, context_obj2_chance], ls='--', color='lightgrey',
         lw=0.75)
ax3.set(xlabel='', title='Context modulation (second object)', yticks=[0, 5, 10, 15], ylim=[0, 15], ylabel='')
ax3.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_context_diff']].groupby('region').mean().sort_values(
    'perc_context_diff', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_context_diff', ax=ax4, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax4.plot(ax1.get_xlim(), [context_onset_chance, context_onset_chance], ls='--', color='lightgrey',
         lw=0.75)
ax4.set(xlabel='', title='Sound identity', yticks=[0, 5, 10], ylim=[0, 10], ylabel='')
ax4.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_reward']].groupby('region').mean().sort_values(
    'perc_reward', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_reward', ax=ax5, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax5.plot(ax1.get_xlim(), [reward_chance, reward_chance], ls='--', color='lightgrey',
         lw=0.75)
ax5.set(xlabel='', title='Outcome modulation', yticks=[0, 10, 20, 30, 40, 50, 60], ylim=[0, 60], ylabel='')
ax5.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.show(block=False)

plt.savefig(path_dict['google_drive_paper_path'] / 'SingleNeurons' / 'perc_sig_neurons.jpg', dpi=600)
plt.savefig(path_dict['google_drive_paper_path'] / 'SingleNeurons' / 'perc_sig_neurons.pdf')


# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5, 1.75), dpi=dpi)
ax1.bar(region_summary.index, region_summary['sum'], color='grey')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set(ylabel='Total number of neurons')

ax2.bar(region_summary.index, region_summary['mean'], yerr=region_summary['sem'],
        color='grey')
ax2.tick_params(axis='x', labelrotation=90)
ax2.set(ylabel='Simultaneously recorded')

sns.despine(trim=False)
plt.tight_layout()

plt.savefig(path_dict['google_drive_paper_path'] / 'SingleNeurons' / 'n_neurons.jpg', dpi=600)
plt.savefig(path_dict['google_drive_paper_path'] / 'SingleNeurons' / 'n_neurons.pdf')

