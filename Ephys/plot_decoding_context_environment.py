# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:27:02 2025

By Guido Meijer
"""


import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
from msvr_functions import paths, load_subjects, figure_style, add_significance
colors, dpi = figure_style()

# Load in data
path_dict = paths()
subjects = load_subjects()
context_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_GLM_position.csv'))
context_df['region'] = context_df['region'].astype(str)
context_df = context_df[context_df['region'] != 'iCA1']
context_df.loc[context_df['region'] == 'dCA1', 'region'] = 'CA1'


def run_ttest(accuracy_series):
    """
    Runs a one-sample t-test on a pandas Series against 0.5.
    Returns NaN if there are fewer than 2 data points.
    """
    if len(accuracy_series) < 2:
        return pd.Series([np.nan, np.nan], index=['statistic', 'pvalue'])
    
    # Run the one-sample t-test against the population mean (popmean) of 0.5
    _, pvalue = stats.ttest_1samp(accuracy_series, popmean=0.5)
    return pd.Series([pvalue], index=['pvalue'])


def run_wilcoxon(accuracy_series):
    """
    Runs a one-sample t-test on a pandas Series against 0.5.
    Returns NaN if there are fewer than 2 data points.
    """
    if len(accuracy_series) < 2:
        return pd.Series([np.nan, np.nan], index=['statistic', 'pvalue'])
    
    # Run the one-sample t-test against the population mean (popmean) of 0.5
    _, pvalue = stats.wilcoxon(accuracy_series - 0.5)
    return pd.Series([pvalue], index=['pvalue'])


# %%

f, axs = plt.subplots(2, 3, figsize=(7, 1.75 * 2), dpi=dpi, sharey=True)
axs = np.concatenate(axs)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'CA1']):
    
    # Do statistics
    this_df = context_df[(context_df['region'] == region)
                         & np.isin(context_df['subject'].values,
                                   subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int))]
    results_df = this_df.groupby('position')['accuracy'].apply(run_wilcoxon).reset_index()
    p_values = results_df['accuracy'].values
    
    # Plot
    sns.lineplot(this_df, x='position', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None, zorder=1)
    axs[i].plot([0, 1500], [0.5, 0.5], ls='--', color='grey', zorder=0)
    axs[i].plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=0, lw=0.5)
    axs[i].plot([900, 900], [0.3, 0.9], ls='--', color='grey', zorder=0, lw=0.5)
    axs[i].text(0, 0.8, region, color=colors[region], ha='left', va='center', fontsize=8,
                fontweight='bold')
    add_significance(results_df['position'].values, p_values, axs[i], y_pos=0.875, alpha=0.05)
    axs[i].set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
               yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90], 
               ylim=[0.3, 0.9], xlabel='', ylabel='')
f.text(0.5, 0.04, 'Position (cm)', ha='center')
f.text(0.06, 0.5, 'Context decoding accuracy (%)', ha='center', va='center', rotation='vertical')
plt.subplots_adjust(left=0.11, bottom=0.12, right=0.98, top=0.9, wspace=0.1, hspace=0.3)
sns.despine(trim=True)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decode_context_GLM_position_near.pdf'))


# %%

f, axs = plt.subplots(2, 3, figsize=(7, 1.75 * 2), dpi=dpi, sharey=True)
axs = np.concatenate(axs)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'CA1']):
    
    # Do statistics
    this_df = context_df[(context_df['region'] == region)
                         & np.isin(context_df['subject'].values,
                                   subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
    results_df = this_df.groupby('position')['accuracy'].apply(run_wilcoxon).reset_index()
    p_values = results_df['accuracy'].values
    
    # Plot
    sns.lineplot(this_df, x='position', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None, zorder=1)
    axs[i].plot([0, 1500], [0.5, 0.5], ls='--', color='grey', zorder=0)
    axs[i].plot([450, 450], [0.3, 0.9], ls='--', color='grey', zorder=0, lw=0.5)
    axs[i].plot([1350, 1350], [0.3, 0.9], ls='--', color='grey', zorder=0, lw=0.5)
    axs[i].text(0, 0.8, region, color=colors[region], ha='left', va='center', fontsize=8,
                fontweight='bold')
    add_significance(results_df['position'].values, p_values, axs[i], y_pos=0.875, alpha=0.05)
    axs[i].set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
               yticks=[0.3, 0.5, 0.7, 0.9], yticklabels=[30, 50, 70, 90], 
               ylim=[0.3, 0.9], xlabel='', ylabel='')
f.text(0.5, 0.04, 'Position (cm)', ha='center')
f.text(0.06, 0.5, 'Context decoding accuracy (%)', ha='center', va='center', rotation='vertical')
plt.subplots_adjust(left=0.11, bottom=0.12, right=0.98, top=0.9, wspace=0.1, hspace=0.3)
sns.despine(trim=True)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decode_context_GLM_position_far.pdf'))
