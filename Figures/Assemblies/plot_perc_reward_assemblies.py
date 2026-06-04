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
assembly_df['sig_exp_obj1'] = assembly_df['p_expectation_obj1'] < 0.05
assembly_df['sig_exp_obj2'] = assembly_df['p_expectation_obj2'] < 0.05
assembly_df['sig_rew_obj1'] = ((assembly_df['p_reward_obj1'] < 0.05) & (assembly_df['p_reward_obj2'] > 0.05))
assembly_df['sig_rew_obj2'] = ((assembly_df['p_reward_obj1'] > 0.05) & (assembly_df['p_reward_obj2'] < 0.05))
assembly_df['sig_ripples'] = assembly_df['p_ripples'] < 0.05
assembly_df['valid_exp_obj1'] = ~np.isnan(assembly_df['p_expectation_obj1'])
assembly_df['valid_exp_obj2'] = ~np.isnan(assembly_df['p_expectation_obj2'])
assembly_df['valid_rew_obj1'] = ~np.isnan(assembly_df['p_reward_obj1'])
assembly_df['valid_rew_obj2'] = ~np.isnan(assembly_df['p_reward_obj2'])
assembly_df['valid_ripples'] = ~np.isnan(assembly_df['p_ripples'])

per_ses_df = assembly_df.groupby(['ses_id', 'region']).agg(
    n_exp_obj1=('valid_exp_obj1', 'sum'),
    sig_exp_obj1=('sig_exp_obj1', 'sum'),
    n_exp_obj2=('valid_exp_obj2', 'sum'),
    sig_exp_obj2=('sig_exp_obj2', 'sum'),
    n_rew_obj1=('valid_rew_obj1', 'sum'),
    sig_rew_obj1=('sig_rew_obj1', 'sum'),
    n_rew_obj2=('valid_rew_obj2', 'sum'),
    sig_rew_obj2=('sig_rew_obj2', 'sum'),
    n_ripples=('valid_ripples', 'sum'),
    sig_ripples=('sig_ripples', 'sum'),
).reset_index()

per_ses_df['perc_sig_exp_obj1'] = (per_ses_df['sig_exp_obj1'] / per_ses_df['n_exp_obj1']) * 100
per_ses_df['perc_sig_exp_obj2'] = (per_ses_df['sig_exp_obj2'] / per_ses_df['n_exp_obj2']) * 100
per_ses_df['perc_sig_rew_obj1'] = (per_ses_df['sig_rew_obj1'] / per_ses_df['n_rew_obj1']) * 100
per_ses_df['perc_sig_rew_obj2'] = (per_ses_df['sig_rew_obj2'] / per_ses_df['n_rew_obj2']) * 100
per_ses_df['perc_sig_ripples'] = (per_ses_df['sig_ripples'] / per_ses_df['n_ripples']) * 100
per_ses_df.replace([np.inf, -np.inf], np.nan, inplace=True)

summary_df = assembly_df.groupby('region').agg(
    n_total_exp_obj1=('valid_exp_obj1', 'sum'),
    n_sig_exp_obj1=('sig_exp_obj1', 'sum'),
    n_total_exp_obj2=('valid_exp_obj2', 'sum'),
    n_sig_exp_obj2=('sig_exp_obj2', 'sum'),
    n_total_rew_obj1=('valid_rew_obj1', 'sum'),
    n_sig_rew_obj1=('sig_rew_obj1', 'sum'),
    n_total_rew_obj2=('valid_rew_obj2', 'sum'),
    n_sig_rew_obj2=('sig_rew_obj2', 'sum'),
    n_total_ripples=('valid_ripples', 'sum'),
    n_sig_ripples=('sig_ripples', 'sum')
)
summary_df['perc_exp_obj1'] = (summary_df['n_sig_exp_obj1'] / summary_df['n_total_exp_obj1']) * 100
summary_df['perc_exp_obj2'] = (summary_df['n_sig_exp_obj2'] / summary_df['n_total_exp_obj2']) * 100
summary_df['perc_rew_obj1'] = (summary_df['n_sig_rew_obj1'] / summary_df['n_total_rew_obj1']) * 100
summary_df['perc_rew_obj2'] = (summary_df['n_sig_rew_obj2'] / summary_df['n_total_rew_obj2']) * 100
summary_df['perc_ripples'] = (summary_df['n_sig_ripples'] / summary_df['n_total_ripples']) * 100
summary_df = summary_df.reset_index()


# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.2 * 2, 1.75), dpi=dpi, sharey=True)

this_order = per_ses_df[['region', 'perc_sig_rew_obj1']].groupby('region').mean().sort_values(
    'perc_sig_rew_obj1', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_rew_obj1', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.plot([-0.5, regions.shape[0] - 0.5], [5, 5], ls='--', color='grey')
ax1.set(ylabel='Significant assemblies (%)', yticks=[0, 10, 20, 30], xlabel='',
        title='Rewarded object 1', ylim=[0, 60], xlim=[-0.5, regions.shape[0] - 0.5])
ax1.tick_params(axis='x', labelrotation=90)

this_order = per_ses_df[['region', 'perc_sig_rew_obj2']].groupby('region').mean().sort_values(
    'perc_sig_rew_obj2', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig_rew_obj2', ax=ax2, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax2.plot([-0.5, regions.shape[0] - 0.5], [5, 5], ls='--', color='grey')
ax2.set(ylabel='Significant assemblies (%)', xlabel='',
        title='Rewarded object 2', xlim=[-0.5, regions.shape[0] - 0.5])
ax2.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'sig_assemblies_reward.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'sig_assemblies_reward.pdf')
plt.show()
