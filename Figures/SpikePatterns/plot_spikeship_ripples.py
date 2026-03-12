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
from msvr_functions import paths, figure_style
colors, dpi = figure_style()

# Load in data
path_dict = paths()
spikeship_df = pd.read_csv(path_dict['google_drive_data_path'] / 'spikeship_ripples_0.75s_8x.csv')

# Plot
f, axs = plt.subplots(2, 6, figsize=(8, 3), dpi=dpi, sharey=False, sharex=True)
plot_df = spikeship_df[(spikeship_df['object'] == 1) & (spikeship_df['time'] > -1) & (spikeship_df['time'] < 2)]
for i, region in enumerate(plot_df['region'].unique()):
    axs[0, i].plot([-1, 2], [0, 0], lw=0.5, ls='--')
    sns.lineplot(data=plot_df[plot_df['region'] == region], x='time', y='contrast_bl', hue='goal', hue_order=[1, 0],
                 palette=[colors['goal'], colors['no-goal']], errorbar='se', err_kws={'lw': 0},
                 legend=None, ax=axs[0, i])
    axs[0, i].set(title=region, xlim=[-1, 2], ylabel='', xlabel='')

plot_df = spikeship_df[(spikeship_df['object'] == 2) & (spikeship_df['time'] > -1) & (spikeship_df['time'] < 2)]
for i, region in enumerate(plot_df['region'].unique()):
    axs[1, i].plot([-1, 2], [0, 0], lw=0.5, ls='--')
    sns.lineplot(data=plot_df[plot_df['region'] == region], x='time', y='contrast_bl', hue='goal', hue_order=[1, 0],
                 palette=[colors['goal'], colors['no-goal']], errorbar='se', err_kws={'lw': 0},
                 legend=None, ax=axs[1, i])
    axs[1, i].set(xlim=[-1, 2], ylabel='', xlabel='')

f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
f.text(0.04, 0.5, 'Spike pattern similarity to ripples', va='center', rotation='vertical')
sns.despine(trim=True)
plt.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.15, wspace=0.7)
plt.savefig(path_dict['google_drive_fig_path'] / 'spikeship_ripples.jpg', dpi=600)
plt.show()