# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 2026

By Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style, load_subjects

colors, dpi = figure_style()

# Load in data
path_dict = paths(sync=False)
subjects = load_subjects()
stats_df = pd.read_csv(join(path_dict['save_path'], 'two_way_anova_residuals.csv'))
stats_df['subject'] = stats_df['subject'].astype(str)
stats_df['date'] = stats_df['date'].astype(str)
stats_df['probe'] = stats_df['probe'].astype(str)

# Basic recording summaries
session_df = stats_df[['subject', 'date', 'probe']].value_counts().reset_index()
print(f'{len(np.unique(session_df["subject"]))} mice')
print(f'{len(np.unique(session_df["date"]))} recording sessions')
print(f'{session_df.shape[0]} probe insertions')
print(f'{stats_df.shape[0]} neurons ({int(session_df["count"].mean())} +- {int(session_df["count"].sem())}, mean +- sem per probe)')

# Filtering & Processing
stats_df = stats_df[stats_df['region'] != 'root']
stats_df = stats_df.dropna(subset=['region'])
stats_df['ses_id'] = stats_df['subject'] + '_' + stats_df['date'] + '_' + stats_df['probe']

# Significant effects (p < 0.05)
stats_df['sig_spatial_bin'] = stats_df['p_spatial_bin'] < 0.05
stats_df['sig_context'] = stats_df['p_context'] < 0.05
stats_df['sig_interaction'] = stats_df['p_interaction'] < 0.05

# Summary statistics per session & region
per_ses_df = stats_df.groupby(['region', 'ses_id']).sum(numeric_only=True)
per_ses_df['n_neurons'] = stats_df.groupby(['region', 'ses_id']).size()
per_ses_df['perc_spatial_bin'] = (per_ses_df['sig_spatial_bin'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_context'] = (per_ses_df['sig_context'] / per_ses_df['n_neurons']) * 100
per_ses_df['perc_interaction'] = (per_ses_df['sig_interaction'] / per_ses_df['n_neurons']) * 100
per_ses_df = per_ses_df.reset_index()

# Overall percentages
print('\nOverall Percentage of Significant Neurons:')
print(f'Spatial Bin: {stats_df["sig_spatial_bin"].mean() * 100:.1f}%')
print(f'Context:     {stats_df["sig_context"].mean() * 100:.1f}%')
print(f'Interaction: {stats_df["sig_interaction"].mean() * 100:.1f}%')

# Summary of neurons per region
region_summary = per_ses_df.groupby('region')['n_neurons'].agg(['mean', 'sum', 'sem'])
region_summary = region_summary.sort_values(by='sum', ascending=False)
print('\nNeuron counts per region:')
print(region_summary)


# %% Figure 1: Percentage of significant neurons per factor across regions
use_xlim = [-0.7, len(per_ses_df['region'].unique()) - 0.3]
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.3 * 3, 2), dpi=dpi, sharey=False)

# 1. Spatial Bin
order_spatial = per_ses_df.groupby('region')['perc_spatial_bin'].mean().sort_values(ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_spatial_bin', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=order_spatial, legend=False)
ax1.plot(use_xlim, [5, 5], ls='--', color='lightgrey', lw=0.75)
ax1.set(ylabel='Significant neurons (%)', xlabel='', title='Spatial Bin', xlim=use_xlim)
ax1.tick_params(axis='x', labelrotation=90)

# 2. Context
order_context = per_ses_df.groupby('region')['perc_context'].mean().sort_values(ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_context', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=order_context, legend=False)
ax2.plot(use_xlim, [5, 5], ls='--', color='lightgrey', lw=0.75)
ax2.set(ylabel='', xlabel='', title='Context', xlim=use_xlim)
ax2.tick_params(axis='x', labelrotation=90)

# 3. Interaction
order_inter = per_ses_df.groupby('region')['perc_interaction'].mean().sort_values(ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_interaction', ax=ax3, hue='region', errorbar='se',
            palette=colors, order=order_inter, legend=False)
ax3.plot(use_xlim, [5, 5], ls='--', color='lightgrey', lw=0.75)
ax3.set(ylabel='', xlabel='', title='Interaction', xlim=use_xlim)
ax3.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()

# Save figure
paper_fig_dir = path_dict['paper_fig_path'] / 'SingleNeurons'
paper_fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(paper_fig_dir / 'perc_two_way_anova_residuals.jpg', dpi=600)
plt.savefig(paper_fig_dir / 'perc_two_way_anova_residuals.pdf')
plt.show(block=False)


# %% Figure 2: Number of neurons per region
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5, 1.75), dpi=dpi)
ax1.bar(region_summary.index, region_summary['sum'], color='grey')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('Total number of neurons', labelpad=1)

ax2.bar(region_summary.index, region_summary['mean'], yerr=region_summary['sem'], color='grey')
ax2.tick_params(axis='x', labelrotation=90)
ax2.set_ylabel('Simultaneously recorded', labelpad=1)

sns.despine(trim=False)
plt.tight_layout(w_pad=0.8)
plt.savefig(paper_fig_dir / 'n_neurons_anova_residuals.jpg', dpi=600)
plt.savefig(paper_fig_dir / 'n_neurons_anova_residuals.pdf')
plt.show(block=False)


# %% Figure 3: Neuronal Overlap Heatmap
overlap_cols = ['sig_spatial_bin', 'sig_context', 'sig_interaction']
overlap_df = stats_df[overlap_cols].astype(int)

# Calculate intersection matrix (% of all neurons)
overlap_matrix = (overlap_df.T @ overlap_df) / stats_df.shape[0] * 100

f, ax = plt.subplots(1, 1, figsize=(3.2, 2.7), dpi=dpi)
sns.heatmap(overlap_matrix, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=['Spatial Bin', 'Context', 'Interaction'],
            yticklabels=['Spatial Bin', 'Context', 'Interaction'],
            ax=ax)
ax.set(title='Factor Overlap (% of all neurons)')
plt.tight_layout()
plt.savefig(paper_fig_dir / 'overlap_two_way_anova_residuals.jpg', dpi=600)
plt.savefig(paper_fig_dir / 'overlap_two_way_anova_residuals.pdf')
plt.show(block=False)


# %% Figure 4: Effect Sizes (Eta-squared) per region
eta_df = stats_df.melt(
    id_vars=['region', 'ses_id'],
    value_vars=['eta_sq_spatial_bin', 'eta_sq_context', 'eta_sq_interaction'],
    var_name='factor',
    value_name='eta_squared'
).dropna()

eta_df['factor'] = eta_df['factor'].replace({
    'eta_sq_spatial_bin': 'Spatial Bin',
    'eta_sq_context': 'Context',
    'eta_sq_interaction': 'Interaction'
})

f, ax = plt.subplots(1, 1, figsize=(4, 2.2), dpi=dpi)
sns.barplot(data=eta_df, x='region', y='eta_squared', hue='factor', ax=ax,
            palette='Set2', errorbar='se')
ax.set(ylabel=r'Effect size ($\eta^2$)', xlabel='', title=r'Effect Size ($\eta^2$) by Region')
ax.tick_params(axis='x', labelrotation=90)
ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(paper_fig_dir / 'eta_squared_anova_residuals.jpg', dpi=600)
plt.savefig(paper_fig_dir / 'eta_squared_anova_residuals.pdf')
plt.show(block=False)
