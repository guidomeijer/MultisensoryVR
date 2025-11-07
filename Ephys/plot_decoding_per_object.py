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
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style, add_significance
colors, dpi = figure_style()

# Load in data
path_dict = paths()
per_obj_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_all_neurons_RF.csv'))

# Drop iCA1 for now
per_obj_df = per_obj_df[per_obj_df['region'] != 'iCA1']
per_obj_df.loc[per_obj_df['region'] == 'dCA1', 'region'] = 'CA1'

def run_ttest(accuracy_series):
    """
    Runs a one-sample t-test on a pandas Series against 0.5.
    Returns NaN if there are fewer than 2 data points.
    """
    if len(accuracy_series) < 2:
        return pd.Series([np.nan, np.nan], index=['statistic', 'pvalue'])
    
    # Run the one-sample t-test against the population mean (popmean) of 0.5
    statistic, pvalue = stats.ttest_1samp(accuracy_series, popmean=0.5)
    return pd.Series([pvalue], index=['pvalue'])


# %%

f, axs = plt.subplots(1, 7, figsize=(8, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'LEC', 'CA1']):
    
    # Do statistics
    this_df = per_obj_df[(per_obj_df['object'] == 1) & (per_obj_df['region'] == region)]
    results_df = this_df.groupby('time')['accuracy'].apply(run_ttest).reset_index()
    #_, p_values = fdrcorrection(results_df['accuracy'].values)
    p_values = results_df['accuracy'].values
    
    axs[i].plot([-2, 2], [0.5, 0.5], ls='--', color='grey')
    axs[i].plot([0,0], [0.4, 0.9], ls='--', color='grey')
    sns.lineplot(this_df, x='time', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None)
    add_significance(results_df['time'].values, p_values, axs[i], y_pos=0.875)
    axs[i].set_title(region)
    if i > 0:
        axs[i].axis('off')
    else:
        axs[i].set(yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], xticks=[], xlabel='',
                   ylabel='Context decoding accuracy', ylim=[0.4, 0.9])
        axs[i].text(-0.28, 0.77, 'Object entry', ha='center', va='center', rotation=90)
        

axs[0].plot([-2, -1], [0.4, 0.4], color='k', lw=1, clip_on=False)
axs[0].text(-1.5, 0.375, '1s', ha='center', va='center')
#axs[0].text(-3.2, 0.6, 'Context decoding accuracy', ha='center', va='center', rotation=90)
plt.subplots_adjust(left=0.05, bottom=None, right=0.99, top=0.85, wspace=0, hspace=None)
sns.despine(bottom=True)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_context_obj1.pdf'))

# %%

f, axs = plt.subplots(1, 7, figsize=(8, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'LEC', 'CA1']):
    
    # Do statistics
    this_df = per_obj_df[(per_obj_df['object'] == 2) & (per_obj_df['region'] == region)]
    results_df = this_df.groupby('time')['accuracy'].apply(run_ttest).reset_index()
    #_, p_values = fdrcorrection(results_df['accuracy'].values)
    p_values = results_df['accuracy'].values
    
    axs[i].plot([-2, 2], [0.5, 0.5], ls='--', color='grey')
    axs[i].plot([0,0], [0.4, 0.9], ls='--', color='grey')
    sns.lineplot(this_df, x='time', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None)
    add_significance(results_df['time'].values, p_values, axs[i], y_pos=0.875)
    axs[i].set_title(region)
    if i > 0:
        axs[i].axis('off')
    else:
        axs[i].set(yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], xticks=[], xlabel='',
                   ylabel='Context decoding accuracy', ylim=[0.4, 0.9])
        axs[i].text(-0.28, 0.77, 'Object entry', ha='center', va='center', rotation=90)
        

axs[0].plot([-2, -1], [0.4, 0.4], color='k', lw=1, clip_on=False)
axs[0].text(-1.5, 0.375, '1s', ha='center', va='center')
#axs[0].text(-3.2, 0.6, 'Context decoding accuracy', ha='center', va='center', rotation=90)
plt.subplots_adjust(left=0.05, bottom=None, right=0.99, top=0.85, wspace=0, hspace=None)
sns.despine(bottom=True)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_context_obj2.pdf'))

# %%

f, axs = plt.subplots(1, 7, figsize=(8, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'LEC', 'CA1']):
    axs[i].plot([-2, 2], [0.5, 0.5], ls='--', color='grey')
    axs[i].plot([0,0], [0.4, 0.8], ls='--', color='grey')
    sns.lineplot(per_obj_df[(per_obj_df['object'] == 3) & (per_obj_df['region'] == region)],
                 x='time', y='accuracy', color=colors[region], errorbar='se',
                 ax=axs[i], err_kws={'lw': 0}, legend=None)
    axs[i].set_title(region)
    axs[i].axis('off')

axs[0].plot([-2, -1], [0.4, 0.4], color='k', lw=1)
axs[0].text(-1.5, 0.375, '1s', ha='center', va='center')
axs[0].plot([-2, -2], [0.4, 0.5], color='k', lw=1)
axs[0].text(-2.4, 0.45, '10%', ha='center', va='center', rotation=90)
axs[0].text(-3.2, 0.6, 'Context decoding accuracy', ha='center', va='center', rotation=90)
plt.subplots_adjust(left=0.05, bottom=None, right=0.99, top=0.85, wspace=0, hspace=None)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decoding_context_obj3.pdf'))
    


