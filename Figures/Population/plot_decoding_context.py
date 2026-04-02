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
from msvr_functions import paths, load_subjects, figure_style, add_significance
colors, dpi = figure_style()

# Settings
MIN_NEURONS = 5
MIN_TRIALS = 2
WIN_NEAR = [900-150, 900]
WIN_FAR = [1350-150, 1350]
REGIONS = ['CA1', 'LEC', 'PERI', 'TEa', 'AUD', 'VIS']

# Load in data
path_dict = paths()
subjects = load_subjects()
context_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_GLM_position.csv'))
context_df['region'] = context_df['region'].astype(str)

# Apply thresholds
context_df = context_df[context_df['n_neurons'] >= MIN_NEURONS]
context_df = context_df[context_df['n_trials'] >= MIN_TRIALS]

# Get average per region in window
context_win_near_df = context_df[(context_df['position'] >= WIN_NEAR[0]) & (context_df['position'] <= WIN_NEAR[1])
                                & np.isin(context_df['subject'].values,
                                       subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int))]

context_win_far_df = context_df[(context_df['position'] >= WIN_FAR[0]) & (context_df['position'] <= WIN_FAR[1])
                                & np.isin(context_df['subject'].values,
                                       subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
context_win_df = pd.concat((context_win_near_df, context_win_far_df))
context_win_df = context_win_df.groupby(['region', 'subject', 'date']).agg({'accuracy': 'mean'}).reset_index()

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
axs = axs.flatten()
for i, region in enumerate(REGIONS):
    
    # Do statistics
    this_df = context_df[(context_df['region'] == region)
                         & np.isin(context_df['subject'].values,
                                   subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int))]
    results_df = this_df.groupby('position')['accuracy'].apply(run_ttest).reset_index()
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
plt.subplots_adjust(left=0.11, bottom=0.12, right=0.98, top=0.9, wspace=0, hspace=0.2)
sns.despine(trim=True)

# Hide top row x-axis 
for i in range(3):
    axs[i].spines['bottom'].set_visible(False)
    axs[i].xaxis.set_ticks_position('none') 
    axs[i].tick_params(labelbottom=False)
    
    
# Hide right column y-axis 
for i in [1, 2, 4, 5]:
    axs[i].spines['left'].set_visible(False)
    axs[i].yaxis.set_ticks_position('none') 
    axs[i].tick_params(labelleft=False)

plt.savefig(join(path_dict['google_drive_fig_path'], 'decode_context_GLM_position_near.pdf'))
plt.savefig(join(path_dict['google_drive_fig_path'], 'decode_context_GLM_position_near.jpg'), dpi=600)


# %%

f, axs = plt.subplots(2, 3, figsize=(7, 1.75 * 2), dpi=dpi, sharey=True)
axs = np.concatenate(axs)
for i, region in enumerate(REGIONS):
    
    # Do statistics
    this_df = context_df[(context_df['region'] == region)
                         & np.isin(context_df['subject'].values,
                                   subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
    results_df = this_df.groupby('position')['accuracy'].apply(run_ttest).reset_index()
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
plt.subplots_adjust(left=0.11, bottom=0.12, right=0.98, top=0.9, wspace=0, hspace=0.2)
sns.despine(trim=True)

# Hide top row x-axis 
for i in range(3):
    axs[i].spines['bottom'].set_visible(False)
    axs[i].xaxis.set_ticks_position('none') 
    axs[i].tick_params(labelbottom=False)
    
    
# Hide right column y-axis 
for i in [1, 2, 4, 5]:
    axs[i].spines['left'].set_visible(False)
    axs[i].yaxis.set_ticks_position('none') 
    axs[i].tick_params(labelleft=False)

plt.savefig(join(path_dict['paper_fig_path'], 'decode_context_GLM_position_far.pdf'))
plt.savefig(join(path_dict['paper_fig_path'], 'decode_context_GLM_position_far.jpg'), dpi=600)
plt.show()

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.3*2, 1.75), dpi=dpi, sharey=True)

plot_df = context_win_df[np.isin(context_win_df['subject'].values,
                         subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int))]
stats_df = plot_df.groupby('region')['accuracy'].apply(run_ttest).reset_index()
stats_df['region'] = pd.Categorical(stats_df['region'], categories=REGIONS, ordered=True)
stats_df = stats_df.sort_values('region')
ax1.plot([-0.5, 6.5], [0.5, 0.5], ls='--', color='grey')
sns.boxplot(plot_df, x='region', y='accuracy', ax=ax1, order=REGIONS, fliersize=0, palette=colors, hue='region',
            linewidth=0.75)
for i, p_value in enumerate(stats_df['accuracy']):
    if p_value < 0.01:
        ax1.text(i, 0.88, '**', fontsize=10, ha='center', va='center')
    elif p_value < 0.05:
        ax1.text(i, 0.88, '*', fontsize=10, ha='center', va='center')
ax1.set(ylabel='Context decoding accuracy (%)', ylim=[0.3, 0.9], xlabel='', yticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        yticklabels=[30, 40, 50, 60, 70, 80, 90], title='Near')
ax1.tick_params(axis='x', labelrotation=90)

plot_df = context_win_df[np.isin(context_win_df['subject'].values,
                         subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
stats_df = plot_df.groupby('region')['accuracy'].apply(run_ttest).reset_index()
stats_df['region'] = pd.Categorical(stats_df['region'], categories=REGIONS, ordered=True)
stats_df = stats_df.sort_values('region')
ax2.plot([-0.5, 6.5], [0.5, 0.5], ls='--', color='grey')
sns.boxplot(plot_df, x='region', y='accuracy', ax=ax2, order=REGIONS, fliersize=0, palette=colors, hue='region',
            linewidth=0.75)
for i, p_value in enumerate(stats_df['accuracy']):
    if p_value < 0.01:
        ax2.text(i, 0.88, '**', fontsize=10, ha='center', va='center')
    elif p_value < 0.05:
        ax2.text(i, 0.88, '*', fontsize=10, ha='center', va='center')
ax2.set(xlabel='', title='Far')
ax2.tick_params(axis='x', labelrotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'context_regions_far_near.pdf')
plt.show()


# %%
f, ax1 = plt.subplots(figsize=(1.5, 1.75), dpi=dpi)

stats_df = context_win_df.groupby('region')['accuracy'].apply(run_ttest).reset_index()
stats_df['region'] = pd.Categorical(stats_df['region'], categories=REGIONS, ordered=True)
stats_df = stats_df.sort_values('region')
ax1.plot([-0.5, 6.5], [0.5, 0.5], ls='--', color='grey')
sns.boxplot(context_win_df, x='region', y='accuracy', ax=ax1, order=REGIONS, fliersize=0, palette=colors, hue='region',
            linewidth=0.5)
for i, p_value in enumerate(stats_df['accuracy']):
    if p_value < 0.01:
        ax1.text(i, 0.88, '**', fontsize=10, ha='center', va='center')
    elif p_value < 0.05:
        ax1.text(i, 0.88, '*', fontsize=10, ha='center', va='center')
ax1.set(ylabel='Context decoding accuracy (%)', ylim=[0.3, 0.9], xlabel='', yticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        yticklabels=[30, 40, 50, 60, 70, 80, 90])
ax1.tick_params(axis='x', labelrotation=90)

sns.despine()
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'context_regions_both.pdf')
plt.show()