# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 09/04/2026
"""
# %%

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

waveform_df = pd.read_csv(path_dict['save_path'] / 'waveform_metrics.csv')
waveform_df['subject'] = waveform_df['subject'].astype(str)
waveform_df['date'] = waveform_df['date'].astype(str)
waveform_df = waveform_df.rename(columns={'unit_id': 'neuron_id'})

# Process
merged_df = pd.merge(neuron_df, waveform_df, on=['subject', 'date', 'probe', 'neuron_id', 'region'])
merged_df['sig_context_obj1'] = merged_df['p_context_obj1'] < 0.05
merged_df['sig_context_obj2'] = merged_df['p_context_obj2'] < 0.05
merged_df['sig_reward'] = merged_df['p_reward'] < 0.05
merged_df['sig_obj_onset'] = merged_df['p_obj_onset'] < 0.05
merged_df = merged_df[merged_df['region'] != 'root']
merged_df['ses_id'] = [f'{merged_df.loc[i, "subject"]}_{merged_df.loc[i, "date"]}' for i in merged_df.index]

# Get chance levels
context_obj1_chance = (np.sum(merged_df['p_context_obj1_shuf'] < 0.05) / merged_df.shape[0]) * 100
context_obj2_chance = (np.sum(merged_df['p_context_obj2_shuf'] < 0.05) / merged_df.shape[0]) * 100
context_onset_chance = (np.sum(merged_df['p_sound_onset_shuf'] < 0.05) / merged_df.shape[0]) * 100
reward_chance = (np.sum(merged_df['p_reward_onset_shuf'] < 0.05) / merged_df.shape[0]) * 100
obj_onset_chance = (np.sum(merged_df['p_obj_onset_shuf'] < 0.05) / merged_df.shape[0]) * 100

# Summary statistics per session
per_ses_df = merged_df.groupby(['neuron_type', 'ses_id']).sum(numeric_only=True)
per_ses_df['n_neurons'] = merged_df.groupby(['neuron_type', 'ses_id']).size()
per_ses_df['perc_context_obj1'] = (per_ses_df['sig_context_obj1'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_context_obj2'] = (per_ses_df['sig_context_obj2'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_reward'] = (per_ses_df['sig_reward'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_obj_onset'] = (per_ses_df['sig_obj_onset'] / per_ses_df['n_neurons']) * 100
per_ses_df = per_ses_df.reset_index()

# %% Plot

f, axs = plt.subplots(2, 2, figsize=(1.8, 1.8), dpi=dpi, sharex=True, sharey='row')
axs = axs.flatten()
_, p = stats.ttest_rel(per_ses_df[per_ses_df['neuron_type'] == 'PYR']['perc_obj_onset'],
                       per_ses_df[per_ses_df['neuron_type'] == 'INT']['perc_obj_onset'])
print(f'Object onset; p = {p:.3f}')
sns.barplot(per_ses_df.sort_values(by='neuron_type', ascending=False), x='neuron_type', y='perc_obj_onset',
            hue='neuron_type', errorbar='se', ax=axs[0], palette=[colors['PYR'], colors['INT']])
axs[0].set(xlabel='', ylim=[0, 50], title='Object onset', ylabel='', yticks=[0, 25, 50])

_, p = stats.ttest_rel(per_ses_df[per_ses_df['neuron_type'] == 'PYR']['perc_reward'],
                       per_ses_df[per_ses_df['neuron_type'] == 'INT']['perc_reward'])
print(f'Reward; p = {p:.3f}')
sns.barplot(per_ses_df.sort_values(by='neuron_type', ascending=False), x='neuron_type', y='perc_reward',
            hue='neuron_type', errorbar='se', ax=axs[1], palette=[colors['PYR'], colors['INT']])
if p < 0.05:
    axs[1].text(0.5, 43, '*', fontsize=12, ha='center')
axs[1].set(ylabel='', xlabel='', ylim=[0, 50], title='Outcome')

_, p = stats.ttest_rel(per_ses_df[per_ses_df['neuron_type'] == 'PYR']['perc_context_obj1'],
                       per_ses_df[per_ses_df['neuron_type'] == 'INT']['perc_context_obj1'])
print(f'Object 1; p = {p:.3f}')
sns.barplot(per_ses_df.sort_values(by='neuron_type', ascending=False), x='neuron_type', y='perc_context_obj1',
            hue='neuron_type', errorbar='se', ax=axs[2], palette=[colors['PYR'], colors['INT']])
axs[2].set(ylabel='', xlabel='', title='Rew. object 1', ylim=[0, 15], yticks=[0, 5, 10, 15])

_, p = stats.ttest_rel(per_ses_df[per_ses_df['neuron_type'] == 'PYR']['perc_context_obj2'],
                       per_ses_df[per_ses_df['neuron_type'] == 'INT']['perc_context_obj2'])
print(f'Object 2; p = {p:.3f}')
sns.barplot(per_ses_df.sort_values(by='neuron_type', ascending=False), x='neuron_type', y='perc_context_obj2',
            hue='neuron_type', errorbar='se', ax=axs[3], palette=[colors['PYR'], colors['INT']])
axs[3].set(ylabel='', xlabel='', title='Rew. object 2')

f.text(0.04, 0.5, 'Significant neurons (%)', va='center', rotation='vertical')

sns.despine(trim=False)
plt.subplots_adjust(left=0.22, bottom=0.1, right=0.95, top=0.9, hspace=0.4, wspace=0.3)
plt.show()
plt.savefig(path_dict['paper_fig_path'] / 'Interneurons' / 'perc_sig_neurons.pdf')