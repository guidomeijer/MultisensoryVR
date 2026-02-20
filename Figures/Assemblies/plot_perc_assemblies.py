# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 20/02/2026
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
assembly_df = pd.read_csv(join(path_dict['save_path'], 'assembly_sig.csv'))
regions = np.unique(assembly_df['region'])

# Calculate percentage significant per session
assembly_df['ses_id'] = assembly_df['subject'] + assembly_df['date']
assembly_df['sig_obj1'] = assembly_df['p_obj1'] < 0.05
assembly_df['sig_obj2'] = assembly_df['p_obj2'] < 0.05
assembly_df['sig_sound'] = assembly_df['p_sound_id'] < 0.05
assembly_df['sig_ripples'] = assembly_df['p_ripples'] < 0.05
assembly_df['log_p_obj1'] = -np.log10(assembly_df['p_obj1'])
assembly_df['log_p_obj2'] = -np.log10(assembly_df['p_obj2'])
assembly_df['log_p_sound'] = -np.log10(assembly_df['p_sound_id'])
assembly_df['log_p_ripples'] = -np.log10(assembly_df['p_ripples'])
per_ses_df = assembly_df[['ses_id', 'region']].groupby(['ses_id', 'region']).size().to_frame()
per_ses_df = per_ses_df.rename(columns={0: 'n_patterns'})
per_ses_df['sig_obj1'] = assembly_df[['ses_id', 'sig_obj1', 'region']].groupby(['ses_id', 'region']).sum()['sig_obj1']
per_ses_df['sig_obj2'] = assembly_df[['ses_id', 'sig_obj2', 'region']].groupby(['ses_id', 'region']).sum()['sig_obj2']
per_ses_df['sig_sound'] = assembly_df[['ses_id', 'sig_sound', 'region']].groupby(['ses_id', 'region']).sum()[
    'sig_sound']
per_ses_df['sig_ripples'] = assembly_df[['ses_id', 'sig_ripples', 'region']].groupby(['ses_id', 'region']).sum()[
    'sig_ripples']
per_ses_df = per_ses_df.reset_index()
per_ses_df['perc_sig_obj1'] = (per_ses_df['sig_obj1'] / per_ses_df['n_patterns']) * 100
per_ses_df['perc_sig_obj2'] = (per_ses_df['sig_obj2'] / per_ses_df['n_patterns']) * 100
per_ses_df['perc_sig_sound'] = (per_ses_df['sig_sound'] / per_ses_df['n_patterns']) * 100
per_ses_df['perc_sig_ripples'] = (per_ses_df['sig_ripples'] / per_ses_df['n_patterns']) * 100

summary_df = assembly_df.groupby('region').size().rename('n_total').to_frame()
summary_df['n_obj1'] = assembly_df.groupby('region').sum()['sig_obj1']
summary_df['n_obj2'] = assembly_df.groupby('region').sum()['sig_obj2']
summary_df['n_ripples'] = assembly_df.groupby('region').sum()['sig_ripples']
summary_df['n_sound'] = assembly_df.groupby('region').sum()['sig_sound']
summary_df['perc_obj1'] = (summary_df['n_obj1'] / summary_df['n_total']) * 100
summary_df['perc_obj2'] = (summary_df['n_obj2'] / summary_df['n_total']) * 100
summary_df['perc_ripples'] = (summary_df['n_ripples'] / summary_df['n_total']) * 100
summary_df['perc_sound'] = (summary_df['n_sound'] / summary_df['n_total']) * 100
summary_df = summary_df.reset_index()

# %% Do statistics

assembly_df = assembly_df[~np.isnan(assembly_df['p_ripples'])]
# Iterate through regions
f, axs = plt.subplots(1, 6, figsize=(7, 1.75), dpi=dpi)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']):
    group = assembly_df[assembly_df['region'] == region]

    # Do Spearman correlation
    # rho, p_val = stats.spearmanr(np.abs(group['t_obj2']), np.abs(group['amp_ripples']))
    # rho, p_val = stats.spearmanr(group['log_p_obj2'], group['log_p_ripples'])
    # rho, p_val = stats.spearmanr(group['z_obj2'], group['z_ripples'])
    rho, p_val = stats.spearmanr(np.abs(group['z_obj2']), np.abs(group['z_ripples']))
    # rho, p_val = stats.pearsonr(group['log_p_obj2'], group['log_p_ripples'])
    print(f'{region}; p = {np.round(p_val, 3)}')

    # Plot
    # axs[i].scatter(np.abs(group['t_obj2']), np.abs(group['amp_ripples']), s=3)
    # axs[i].scatter(group['log_p_obj2'], group['log_p_ripples'], s=3)
    # axs[i].scatter(group['z_obj2'], group['z_ripples'], s=3)
    axs[i].scatter(np.abs(group['z_obj2']), np.abs(group['z_ripples']), s=3)
    axs[i].set(title=f'{region}; p = {np.round(p_val, 3)}')
    if i == 0:
        axs[i].set(ylabel='Ripples (-log10[p])')

    # if p_val < 0.05:
    #    axs[i].text(4.5, 10, f'p = {np.round(p_val,3)}', ha='center', va='center')
    #    axs[i].text(4.5, 8.5, '**', fontsize=12, ha='center', va='center')

f.text(0.5, 0.04, 'Second object (-log10[p])', ha='center')

sns.despine(trim=False)
plt.subplots_adjust(left=0.06, bottom=0.2, top=0.88, right=0.98)

# %%

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75 * 3, 2), dpi=dpi, sharey=True)

this_order = per_ses_df[['region', 'perc_sig_obj1']].groupby('region').mean().sort_values(
    'perc_sig_obj1', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_obj1', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.plot([-0.5, regions.shape[0] - 0.5], [5, 5], ls='--', color='grey')
ax1.set(ylabel='Significant assemblies (%)', yticks=[0, 10, 20, 30, 40], xlabel='',
        title='Object 1', ylim=[0, 40], xlim=[-0.5, regions.shape[0] - 0.5])
ax1.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_sig_obj2']].groupby('region').mean().sort_values(
    'perc_sig_obj2', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_obj2', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax2.plot([-0.5, regions.shape[0] - 0.5], [5, 5], ls='--', color='grey')
ax2.set(ylabel='Significant assemblies (%)', xlabel='',
        title='Object 2', xlim=[-0.5, regions.shape[0] - 0.5])
ax2.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_sig_sound']].groupby('region').mean().sort_values(
    'perc_sig_sound', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_sound', ax=ax3, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax3.plot([-0.5, regions.shape[0] - 0.5], [5, 5], ls='--', color='grey')
ax3.set(ylabel='Significant assemblies (%)', xlabel='',
        title='Sound onset', xlim=[-0.5, regions.shape[0] - 0.5])
ax3.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'sig_assemblies.jpg', dpi=600)

f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
this_order = per_ses_df[['region', 'perc_sig_ripples']].groupby('region').mean().sort_values(
    'perc_sig_ripples', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_ripples', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.plot([-0.5, regions.shape[0] - 0.5], [5, 5], ls='--', color='grey')
ax1.set(ylabel='Significant assemblies (%)', xlabel='', yticks=np.arange(0, 101, 20),
        title='Ripples', xlim=[-0.5, regions.shape[0] - 0.5])
ax1.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.show()
plt.savefig(path_dict['google_drive_fig_path'] / 'sig_assemblies_ripples.jpg', dpi=600)

# %% Calculate overlap
assembly_df['sig_both_objs'] = assembly_df['sig_obj1'] & assembly_df['sig_obj2']
assembly_df['sig_obj1_ripple'] = assembly_df['sig_obj1'] & assembly_df['sig_ripples']
assembly_df['sig_obj2_ripple'] = assembly_df['sig_obj2'] & assembly_df['sig_ripples']
assembly_df['sig_all'] = assembly_df['sig_obj1'] & assembly_df['sig_obj2'] & assembly_df['sig_ripples']

overlap_ses = assembly_df.groupby(['ses_id', 'region']).agg(
    n_patterns=('sig_obj1', 'size'),
    n_obj1=('sig_obj1', 'sum'),
    n_obj2=('sig_obj2', 'sum'),
    n_ripples=('sig_ripples', 'sum'),
    n_obj1_ripple=('sig_obj1_ripple', 'sum'),
    n_obj2_ripple=('sig_obj2_ripple', 'sum')
).reset_index()

overlap_ses['perc_obj1_ripple'] = (overlap_ses['n_obj1_ripple'] / overlap_ses['n_patterns']) * 100
overlap_ses['perc_obj2_ripple'] = (overlap_ses['n_obj2_ripple'] / overlap_ses['n_patterns']) * 100

# Plot overlap
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2), dpi=dpi, sharey=True)

this_order = overlap_ses.groupby('region')['perc_obj1_ripple'].mean().sort_values(ascending=False).index.values
sns.barplot(data=overlap_ses, x='region', y='perc_obj1_ripple', ax=ax1, hue='region',
            palette=colors, order=this_order, errorbar='se')
ax1.set(ylabel='Overlap (%)', title='Obj 1 & Ripples', xlabel='')
ax1.tick_params(axis='x', labelrotation=90)

this_order = overlap_ses.groupby('region')['perc_obj2_ripple'].mean().sort_values(ascending=False).index.values
sns.barplot(data=overlap_ses, x='region', y='perc_obj2_ripple', ax=ax2, hue='region',
            palette=colors, order=this_order, errorbar='se')
ax2.set(ylabel='', title='Obj 2 & Ripples', xlabel='')
ax2.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.show()