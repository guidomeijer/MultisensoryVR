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
#assembly_df['ses_id'] = assembly_df['subject'] + assembly_df['date']
assembly_df['ses_id'] = assembly_df['date']
assembly_df['sig_obj1'] = assembly_df['p_obj1'] < 0.05
assembly_df['sig_obj2'] = assembly_df['p_obj2'] < 0.05
assembly_df['sig_sound'] = assembly_df['p_sound_id'] < 0.05
assembly_df['sig_ripples'] = assembly_df['p_ripples'] < 0.05
assembly_df['valid_obj1'] = ~np.isnan(assembly_df['p_obj1'])
assembly_df['valid_obj2'] = ~np.isnan(assembly_df['p_obj2'])
assembly_df['valid_sound'] = ~np.isnan(assembly_df['p_sound_id'])
assembly_df['valid_ripples'] = ~np.isnan(assembly_df['p_ripples'])
assembly_df['log_p_obj1'] = -np.log10(assembly_df['p_obj1'])
assembly_df['log_p_obj2'] = -np.log10(assembly_df['p_obj2'])
assembly_df['log_p_sound'] = -np.log10(assembly_df['p_sound_id'])
assembly_df['log_p_ripples'] = -np.log10(assembly_df['p_ripples'])

# Add columns for stacked plot
assembly_df['sig_ripples_only'] = (assembly_df['sig_ripples']) & (~assembly_df['sig_obj2'])
assembly_df['sig_obj2_only'] = (~assembly_df['sig_ripples']) & (assembly_df['sig_obj2'])
assembly_df['sig_both'] = (assembly_df['sig_ripples']) & (assembly_df['sig_obj2'])
assembly_df['valid_both'] = (assembly_df['valid_ripples']) & (assembly_df['valid_obj2'])

per_ses_df = assembly_df.groupby(['ses_id', 'region']).agg(
    n_obj1=('valid_obj1', 'sum'),
    sig_obj1=('sig_obj1', 'sum'),
    n_obj2=('valid_obj2', 'sum'),
    sig_obj2=('sig_obj2', 'sum'),
    n_sound=('valid_sound', 'sum'),
    sig_sound=('sig_sound', 'sum'),
    n_ripples=('valid_ripples', 'sum'),
    sig_ripples=('sig_ripples', 'sum'),
    n_both=('valid_both', 'sum'),
    sig_ripples_only=('sig_ripples_only', 'sum'),
    sig_obj2_only=('sig_obj2_only', 'sum'),
    sig_both=('sig_both', 'sum')
).reset_index()

per_ses_df['perc_sig_obj1'] = (per_ses_df['sig_obj1'] / per_ses_df['n_obj1']) * 100
per_ses_df['perc_sig_obj2'] = (per_ses_df['sig_obj2'] / per_ses_df['n_obj2']) * 100
per_ses_df['perc_sig_sound'] = (per_ses_df['sig_sound'] / per_ses_df['n_sound']) * 100
per_ses_df['perc_sig_ripples'] = (per_ses_df['sig_ripples'] / per_ses_df['n_ripples']) * 100
per_ses_df['perc_sig_ripples_only'] = (per_ses_df['sig_ripples_only'] / per_ses_df['n_both']) * 100
per_ses_df['perc_sig_obj2_only'] = (per_ses_df['sig_obj2_only'] / per_ses_df['n_both']) * 100
per_ses_df['perc_sig_both'] = (per_ses_df['sig_both'] / per_ses_df['n_both']) * 100
per_ses_df.replace([np.inf, -np.inf], np.nan, inplace=True)

summary_df = assembly_df.groupby('region').agg(
    n_total_obj1=('valid_obj1', 'sum'),
    n_sig_obj1=('sig_obj1', 'sum'),
    n_total_obj2=('valid_obj2', 'sum'),
    n_sig_obj2=('sig_obj2', 'sum'),
    n_total_sound=('valid_sound', 'sum'),
    n_sig_sound=('sig_sound', 'sum'),
    n_total_ripples=('valid_ripples', 'sum'),
    n_sig_ripples=('sig_ripples', 'sum')
)
summary_df['perc_obj1'] = (summary_df['n_sig_obj1'] / summary_df['n_total_obj1']) * 100
summary_df['perc_obj2'] = (summary_df['n_sig_obj2'] / summary_df['n_total_obj2']) * 100
summary_df['perc_ripples'] = (summary_df['n_sig_ripples'] / summary_df['n_total_ripples']) * 100
summary_df['perc_sound'] = (summary_df['n_sig_sound'] / summary_df['n_total_sound']) * 100
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
    rho, p_val = stats.pearsonr(np.abs(group['z_obj2']), np.abs(group['z_ripples']))
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

# %% Stacked bar plot
f, ax = plt.subplots(figsize=(2, 2), dpi=dpi)
stacked_df = per_ses_df.groupby('region')[['perc_sig_ripples_only', 'perc_sig_obj2_only', 'perc_sig_both']].mean()
stacked_df = stacked_df.fillna(0)
stacked_df['total'] = stacked_df.sum(axis=1)
stacked_df = stacked_df.sort_values('total', ascending=False).drop(columns='total')
stacked_df.rename(columns={'perc_sig_ripples_only': 'Ripples only',
                           'perc_sig_obj2_only': 'Object 2 only',
                           'perc_sig_both': 'Both'}, inplace=True)
stacked_df.plot(kind='bar', stacked=True, ax=ax)
ax.set(ylabel='Significant assemblies (%)', xlabel='', title='Ripples & Object 2')
ax.legend(frameon=False, bbox_to_anchor=(1, 1))
ax.tick_params(axis='x', labelrotation=90)
sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'sig_assemblies_stacked.jpg', dpi=600)
plt.show()
