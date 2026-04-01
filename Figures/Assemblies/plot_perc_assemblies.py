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
per_ses_df['perc_obj2_ripple'] = (per_ses_df['sig_both'] / per_ses_df['sig_obj2']) * 100
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


# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.2 * 2, 1.75), dpi=dpi, sharey=True)

this_order = per_ses_df[['region', 'perc_sig_obj1']].groupby('region').mean().sort_values(
    'perc_sig_obj1', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_obj1', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.plot([-0.5, regions.shape[0] - 0.5], [5, 5], ls='--', color='grey')
ax1.set(ylabel='Significant assemblies (%)', yticks=[0, 10, 20, 30], xlabel='',
        title='Rewarded object 1', ylim=[0, 30], xlim=[-0.5, regions.shape[0] - 0.5])
ax1.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_sig_obj2']].groupby('region').mean().sort_values(
    'perc_sig_obj2', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_obj2', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax2.plot([-0.5, regions.shape[0] - 0.5], [5, 5], ls='--', color='grey')
ax2.set(ylabel='Significant assemblies (%)', xlabel='',
        title='Rewarded object 2', xlim=[-0.5, regions.shape[0] - 0.5])
ax2.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'sig_assemblies.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'sig_assemblies.pdf')

f, ax1 = plt.subplots(figsize=(1.5, 1.75), dpi=dpi)
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
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' /'sig_assemblies_ripples.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' /'sig_assemblies_ripples.pdf')

f, ax1 = plt.subplots(figsize=(1.5, 1.75), dpi=dpi)
this_order = per_ses_df[['region', 'perc_obj2_ripple']].groupby('region').mean().sort_values(
    'perc_obj2_ripple', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_obj2_ripple', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.set(ylabel='Ripple modulated (%)', xlabel='', yticks=np.arange(0, 101, 20),
        title='Obj 2 assemblies', xlim=[-0.5, regions.shape[0] - 0.5])
ax1.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'sig_assemblies_obj2_ripples.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'sig_assemblies_obj2_ripples.pdf')
plt.show()
