# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 2026 by Guido Meijer

Plot percentages of significant neurons from the Two-Way ANOVA
mixed selectivity analysis: Object, Goal (reward context), and
their Interaction per brain region.
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
stats_df = pd.read_csv(join(path_dict['save_path'], 'mixed_selectivity_anova.csv'))
stats_df['subject'] = stats_df['subject'].astype(str)
stats_df['date'] = stats_df['date'].astype(str)

# Print summary
session_df = stats_df[['subject', 'date', 'probe']].value_counts().reset_index()
print(f'{len(np.unique(session_df["subject"]))} mice')
print(f'{len(np.unique(session_df["date"]))} recording sessions')
print(f'{session_df.shape[0]} probe insertions')
print(f'{stats_df.shape[0]} neurons ({int(session_df["count"].mean())} '
      f'+- {int(session_df["count"].sem())}, mean +- sem per probe)')

# Exclude neurons outside target regions
stats_df = stats_df[stats_df['region'] != 'root']

# Flag significant neurons (α = 0.05)
stats_df['sig_object'] = stats_df['p_object'] < 0.05
stats_df['sig_goal'] = stats_df['p_goal'] < 0.05
stats_df['sig_interaction'] = stats_df['p_interaction'] < 0.05

# Create session identifier
stats_df['ses_id'] = [f'{stats_df.loc[i, "subject"]}_{stats_df.loc[i, "date"]}'
                      for i in stats_df.index]

# Merge perirhinal sub-areas
stats_df.loc[stats_df['region'] == 'PERI 36', 'region'] = 'PERI'
stats_df.loc[stats_df['region'] == 'PERI 35', 'region'] = 'PERI'

# Chance level (5 %)
CHANCE = 5

# Summary statistics per session
per_ses_df = stats_df.groupby(['region', 'ses_id']).sum(numeric_only=True)
per_ses_df['n_neurons'] = stats_df.groupby(['region', 'ses_id']).size()
per_ses_df['perc_object'] = (per_ses_df['sig_object'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_goal'] = (per_ses_df['sig_goal'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_interaction'] = (per_ses_df['sig_interaction'] / per_ses_df['n_neurons']) * 100
per_ses_df = per_ses_df.reset_index()

# Print overall percentages
print(f'\nOverall significant neurons:')
print(f'  Object identity:  {stats_df["sig_object"].mean()*100:.1f}%')
print(f'  Goal (context):   {stats_df["sig_goal"].mean()*100:.1f}%')
print(f'  Interaction:      {stats_df["sig_interaction"].mean()*100:.1f}%')


# %% Plot percentage significant neurons per region for each ANOVA factor
use_xlim = [-0.7, 5.5]
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.1*3, 2), dpi=dpi, sharey=False)

# ── Object identity ──────────────────────────────────────────────────────────
this_order = per_ses_df[['region', 'perc_object']].groupby('region').mean().sort_values(
    'perc_object', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_object', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.plot(use_xlim, [CHANCE, CHANCE], ls='--', color='lightgrey', lw=0.75)
ax1.set(ylabel='Significant neurons (%)', xlabel='',
        title='Object identity', xlim=use_xlim)
ax1.tick_params(axis='x', labelrotation=90)

# ── Goal (reward context) ────────────────────────────────────────────────────
this_order = per_ses_df[['region', 'perc_goal']].groupby('region').mean().sort_values(
    'perc_goal', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_goal', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax2.plot(use_xlim, [CHANCE, CHANCE], ls='--', color='lightgrey', lw=0.75)
ax2.set(ylabel='', xlabel='',
        title='Reward context', xlim=use_xlim)
ax2.tick_params(axis='x', labelrotation=90)

# ── Interaction (conjunctive / mixed selectivity) ─────────────────────────────
this_order = per_ses_df[['region', 'perc_interaction']].groupby('region').mean().sort_values(
    'perc_interaction', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_interaction', ax=ax3, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax3.plot(use_xlim, [CHANCE, CHANCE], ls='--', color='lightgrey', lw=0.75)
ax3.set(ylabel='', xlabel='',
        title='Interaction\n(mixed selectivity)', xlim=use_xlim)
ax3.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.show(block=False)

plt.savefig(path_dict['paper_fig_path'] / 'SingleNeurons' / 'perc_mixed_selectivity.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'SingleNeurons' / 'perc_mixed_selectivity.pdf')


# %% Overlap heatmap: which neurons are significant for multiple factors?
overlap_cols = ['sig_object', 'sig_goal', 'sig_interaction']
overlap_df = stats_df[overlap_cols].astype(int)

overlap_matrix = (overlap_df.T @ overlap_df) / stats_df.shape[0] * 100

f, ax1 = plt.subplots(1, 1, figsize=(3, 2.5), dpi=dpi)
sns.heatmap(overlap_matrix, annot=True, fmt='.1f', cmap='Reds',
            xticklabels=['Object', 'Goal', 'Interaction'],
            yticklabels=['Object', 'Goal', 'Interaction'])
ax1.set(title='Neuronal overlap (% of all neurons)')
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'SingleNeurons' / 'overlap_mixed_selectivity.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'SingleNeurons' / 'overlap_mixed_selectivity.pdf')
plt.show()
