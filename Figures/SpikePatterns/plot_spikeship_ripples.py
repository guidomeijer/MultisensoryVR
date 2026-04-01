# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 12/03/2026
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from msvr_functions import paths, figure_style, add_significance
colors, dpi = figure_style()

# Load in data
path_dict = paths()
spikeship_df = pd.read_csv(path_dict['google_drive_data_path'] / 'spikeship_ripples_0.75s_8x.csv')

# Plot
#y_extend = {'PERI': 0.001, 'TEa': 0.002, 'VIS': 0.005, 'LEC': 0.004, 'CA1': 0.001, 'AUD': 0.01}
f, axs = plt.subplots(2, 6, figsize=(6, 3), dpi=dpi, sharey=False, sharex=True)
plot_df = spikeship_df[(spikeship_df['object'] == 1) & (spikeship_df['time'] > -1) & (spikeship_df['time'] < 2)]
for i, region in enumerate(plot_df['region'].unique()):
    reg_data = plot_df[plot_df['region'] == region]
    axs[0, i].plot([-1, 2], [0, 0], lw=0.5, ls='--', color='grey')
    sns.lineplot(data=reg_data, x='time', y='contrast_bl', hue='goal', hue_order=[1, 0],
                 palette=[colors['goal'], colors['no-goal']], errorbar='se', err_kws={'lw': 0},
                 legend=None, ax=axs[0, i])
    axs[0, i].plot([-1.1, -1.1], [0, 0.001], color='k', clip_on=False)
    if i == 0:
        axs[0, i].text(-1.3, 0.0005, 0.001, ha='center', va='center', rotation=90)
    
    # Statistical test per timepoint
    times = np.sort(reg_data['time'].unique())
    p_values_goal = [ttest_1samp(reg_data[(reg_data['time'] == t) & (reg_data['goal'] == 1)]['contrast_bl'],
                                0, nan_policy='omit').pvalue for t in times]
    p_values_nogoal = [ttest_1samp(reg_data[(reg_data['time'] == t) & (reg_data['goal'] == 0)]['contrast_bl'],
                                  0, nan_policy='omit').pvalue for t in times]
    axs[0, i].set(xlim=[-1, 2], ylabel='', xlabel='')
    axs[0, i].set_title(region, weight='bold')

    ymin, ymax = axs[0, i].get_ylim()
    mag = max(abs(ymin), abs(ymax))
    axs[0, i].set_ylim(-mag, mag)
    axs[0, i].plot([0, 0], [-1, 1], lw=0.5, ls='--', color='grey')

    add_significance(times, np.array(p_values_goal), axs[0, i], color=colors['goal'],
                     y_pos=mag + (mag*0.03))
    add_significance(times, np.array(p_values_nogoal), axs[0, i], color=colors['no-goal'])

    axs[0, i].axis('off')

plot_df = spikeship_df[(spikeship_df['object'] == 2) & (spikeship_df['time'] > -1) & (spikeship_df['time'] < 2)]
for i, region in enumerate(plot_df['region'].unique()):
    reg_data = plot_df[plot_df['region'] == region]
    axs[1, i].plot([-1, 2], [0, 0], lw=0.5, ls='--', color='grey')
    sns.lineplot(data=reg_data, x='time', y='contrast_bl', hue='goal', hue_order=[1, 0],
                 palette=[colors['goal'], colors['no-goal']], errorbar='se', err_kws={'lw': 0},
                 legend=None, ax=axs[1, i])
    axs[1, i].plot([-1.1, -1.1], [0, 0.001], color='k', clip_on=False)

    # Statistical test per timepoint
    times = np.sort(reg_data['time'].unique())
    p_values_goal = [ttest_1samp(reg_data[(reg_data['time'] == t) & (reg_data['goal'] == 1)]['contrast_bl'],
                                0, nan_policy='omit').pvalue for t in times]
    p_values_nogoal = [ttest_1samp(reg_data[(reg_data['time'] == t) & (reg_data['goal'] == 0)]['contrast_bl'],
                                  0, nan_policy='omit').pvalue for t in times]

    axs[1, i].set(xlim=[-1, 2], ylabel='', xlabel='')
    ymin, ymax = axs[1, i].get_ylim()
    mag = max(abs(ymin), abs(ymax))
    axs[1, i].set_ylim(-mag, mag)
    axs[1, i].plot([0, 0], [-1, 1], lw=0.5, ls='--', color='grey')

    if i == 0:
        axs[1, i].text(-1.3, 0.0005, 0.001, ha='center', va='center', rotation=90)
        axs[1, i].plot([-1, 0], [-mag, -mag], color='k', clip_on=False)
        axs[1, i].text(-0.5, -mag - 0.0002, '1s', ha='center', va='center')
        axs[1, i].text(0.15, -0.0017, 'Object entry', rotation='vertical')

    add_significance(times, np.array(p_values_goal), axs[1, i], color=colors['goal'],
                     y_pos=mag + (mag*0.03))
    add_significance(times, np.array(p_values_nogoal), axs[1, i], color=colors['no-goal'])

    axs[1, i].axis('off')

#f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
f.text(0.02, 0.5, 'Spike pattern similarity to ripples', va='center', rotation='vertical')
sns.despine(trim=True)
plt.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.15, wspace=0.3, hspace=0)
plt.savefig(path_dict['paper_fig_path'] / 'SpikePatterns' / 'spikeship_ripples.jpg', dpi=600)
plt.show()