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
neuron_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
neuron_df['subject'] = neuron_df['subject'].astype(str)
neuron_df['date'] = neuron_df['date'].astype(str)
neuron_df.loc[neuron_df['region'] == 'vCA1', 'region'] = 'iCA1'

waveform_df = pd.read_csv(path_dict['save_path'] / 'waveform_metrics.csv')
waveform_df['subject'] = waveform_df['subject'].astype(str)
waveform_df['date'] = waveform_df['date'].astype(str)
waveform_df = waveform_df.rename(columns={'unit_id': 'neuron_id'})

# Merge dataframe
all_stats_df = pd.merge(neuron_df, waveform_df, on=['subject', 'date', 'probe', 'region', 'neuron_id'])

# Narrow spiking neurons
stats_df = all_stats_df[all_stats_df['neuron_type'] == 'INT'].copy()
stats_df['sig_context_obj1'] = stats_df['p_context_obj1'] < 0.05
stats_df['sig_context_obj2'] = stats_df['p_context_obj2'] < 0.05
stats_df['sig_context_diff'] = stats_df['p_sound_diff'] < 0.05
stats_df['sig_reward'] = stats_df['p_reward'] < 0.05
stats_df['sig_obj_onset'] = stats_df['p_obj_onset'] < 0.05
stats_df = stats_df[stats_df['region'] != 'root']
stats_df['ses_id'] = [f'{stats_df.loc[i, "subject"]}_{stats_df.loc[i, "date"]}' for i in stats_df.index]
per_ses_ns = stats_df.groupby(['region', 'ses_id']).sum(numeric_only=True)
per_ses_ns['n_neurons'] = stats_df.groupby(['region', 'ses_id']).size()
per_ses_ns['perc_context_obj1'] = (per_ses_ns['sig_context_obj1'] / per_ses_ns['n_neurons']) * 100
per_ses_ns['perc_context_obj2'] = (per_ses_ns['sig_context_obj2'] / per_ses_ns['n_neurons']) * 100
per_ses_ns['perc_context_onset'] = (per_ses_ns['sig_context_diff'] / per_ses_ns['n_neurons']) * 100
per_ses_ns['perc_reward'] = (per_ses_ns['sig_reward'] / per_ses_ns['n_neurons']) * 100
per_ses_ns['perc_obj_onset'] = (per_ses_ns['sig_obj_onset'] / per_ses_ns['n_neurons']) * 100
per_ses_ns = per_ses_ns.reset_index()
per_ses_ns['neuron_type'] = 'INT'

# Regular spiking neurons
stats_df = all_stats_df[all_stats_df['neuron_type'] == 'PYR'].copy()
stats_df['sig_context_obj1'] = stats_df['p_context_obj1'] < 0.05
stats_df['sig_context_obj2'] = stats_df['p_context_obj2'] < 0.05
stats_df['sig_context_onset'] = stats_df['p_sound_diff'] < 0.05
stats_df['sig_reward'] = stats_df['p_reward'] < 0.05
stats_df['sig_obj_onset'] = stats_df['p_obj_onset'] < 0.05
stats_df = stats_df[stats_df['region'] != 'root']
stats_df['ses_id'] = [f'{stats_df.loc[i, "subject"]}_{stats_df.loc[i, "date"]}' for i in stats_df.index]
per_ses_rs = stats_df.groupby(['region', 'ses_id']).sum(numeric_only=True)
per_ses_rs['n_neurons'] = stats_df.groupby(['region', 'ses_id']).size()
per_ses_rs['perc_context_obj1'] = (per_ses_rs['sig_context_obj1'] / per_ses_rs['n_neurons']) * 100
per_ses_rs['perc_context_obj2'] = (per_ses_rs['sig_context_obj2'] / per_ses_rs['n_neurons']) * 100
per_ses_rs['perc_context_onset'] = (per_ses_rs['sig_context_onset'] / per_ses_rs['n_neurons']) * 100
per_ses_rs['perc_reward'] = (per_ses_rs['sig_reward'] / per_ses_rs['n_neurons']) * 100
per_ses_rs['perc_obj_onset'] = (per_ses_rs['sig_obj_onset'] / per_ses_rs['n_neurons']) * 100
per_ses_rs = per_ses_rs.reset_index()
per_ses_rs['neuron_type'] = 'PYR'

# Merge
per_ses_df = pd.concat((per_ses_ns, per_ses_rs))

# Get chance levels
context_obj1_chance = (np.sum(all_stats_df['p_context_obj1_shuf'] < 0.05) / all_stats_df.shape[0]) * 100
context_obj2_chance = (np.sum(all_stats_df['p_context_obj2_shuf'] < 0.05) / all_stats_df.shape[0]) * 100
context_onset_chance = (np.sum(all_stats_df['p_sound_onset_shuf'] < 0.05) / all_stats_df.shape[0]) * 100
reward_chance = (np.sum(all_stats_df['p_reward_onset_shuf'] < 0.05) / all_stats_df.shape[0]) * 100
obj_onset_chance = (np.sum(all_stats_df['p_obj_onset_shuf'] < 0.05) / all_stats_df.shape[0]) * 100

# %%
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(7, 4), dpi=dpi, sharey=False)

this_order = per_ses_df[['region', 'perc_obj_onset']].groupby('region').mean().sort_values(
    'perc_obj_onset', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_obj_onset', ax=ax1, hue='neuron_type', errorbar='se',
            order=this_order)
ax1.plot(ax1.get_xlim(), [obj_onset_chance, obj_onset_chance], ls='--', color='lightgrey',
         lw=0.75)
ax1.set(ylabel='Significant neurons (%)',  yticks=[0, 20, 40, 60, 80, 100], xlabel='',
        title='Object modulation', ylim=[0, 100])
ax1.legend(title='')
ax1.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_context_obj1']].groupby('region').mean().sort_values(
    'perc_context_obj1', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_context_obj1', ax=ax2, hue='neuron_type', errorbar='se',
            order=this_order, legend=None)
ax2.plot(ax1.get_xlim(), [context_obj1_chance, context_obj1_chance], ls='--', color='lightgrey',
         lw=0.75)
#ax2.set(xlabel='', title='Context first landmark', yticks=[0, 1, 2, 3, 4, 5, 6], ylim=[0, 6], ylabel='')
ax2.set(xlabel='', title='Context modulation (first object)', yticks=[0, 5, 10, 15, 20, 25], ylim=[0, 25], ylabel='')
ax2.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_context_obj2']].groupby('region').mean().sort_values(
    'perc_context_obj2', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_context_obj2', ax=ax3, hue='neuron_type', errorbar='se',
            order=this_order, legend=None)
ax3.plot(ax1.get_xlim(), [context_obj2_chance, context_obj2_chance], ls='--', color='lightgrey',
         lw=0.75)
ax3.set(xlabel='', title='Context modulation (second object)', yticks=[0, 5, 10, 15, 20, 25], ylim=[0, 25], ylabel='')
ax3.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_context_onset']].groupby('region').mean().sort_values(
    'perc_context_onset', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_context_onset', ax=ax4, hue='neuron_type', errorbar='se',
            order=this_order, legend=None)
ax4.plot(ax1.get_xlim(), [context_onset_chance, context_onset_chance], ls='--', color='lightgrey',
         lw=0.75)
ax4.set(xlabel='', title='Sound identity', yticks=[0, 5, 10], ylim=[0, 10], ylabel='')
ax4.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_reward']].groupby('region').mean().sort_values(
    'perc_reward', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_reward', ax=ax5, hue='neuron_type', errorbar='se',
            order=this_order, legend=None)
ax5.plot(ax1.get_xlim(), [reward_chance, reward_chance], ls='--', color='lightgrey',
         lw=0.75)
ax5.set(xlabel='', title='Outcome modulation', yticks=[0, 10, 20, 30, 40, 50, 60], ylim=[0, 60], ylabel='')
ax5.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.show(block=False)

plt.savefig(join(path_dict['google_drive_fig_path'], 'perc_sig_neurons_type.jpg'), dpi=600)


