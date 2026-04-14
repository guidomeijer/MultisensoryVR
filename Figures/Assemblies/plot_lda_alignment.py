# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 13/04/2026
"""
# %%

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style, add_significance
colors, dpi = figure_style()
path_dict = paths()

# Settings
REGIONS = ['CA1', 'LEC', 'PERI', 'TEa', 'AUD', 'VIS']

# Load in data
lda_df = pd.read_csv(path_dict['google_drive_data_path'] / 'lda_alignment_pre_object.csv')
lda_df = lda_df[(lda_df['time_ax'] >= -1) & (lda_df['time_ax'] <= 1)]

# %% Plot

f, axs = plt.subplots(2, 6, figsize=(7, 3.5), dpi=dpi, sharex=True, sharey='row')
for r, region in enumerate(REGIONS):

    # Perform t-test per timepoint
    p_values = np.full(np.unique(lda_df['time_ax']).shape[0], np.nan)
    for k, t in enumerate(lda_df['time_ax'].unique()):
        data_real = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 1) & (lda_df['time_ax'] == t), 'lda_align_bl']
        data_shuf = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 1) & (lda_df['time_ax'] == t), 'lda_align_bl_shuf']
        _, p_values[k] = ttest_rel(data_real, data_shuf)

    #axs[r].plot([-1, 1], [0, 0], color='darkgrey', ls='--')
    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 1)], x='time_ax', y='lda_align_bl_shuf',
                 errorbar='se', ax=axs[0, r], err_kws={'lw': 0}, color='grey', zorder=0)
    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 1)], x='time_ax', y='lda_align_bl',
                 errorbar='se', ax=axs[0, r], err_kws={'lw': 0}, zorder=1, color=colors[region])
    add_significance(np.unique(lda_df['time_ax']), p_values, ax=axs[0, r], color=colors[region])
    axs[0, r].set(ylabel='', xlabel='', title=region, xlim=[-1, 1])

    # Perform t-test per timepoint
    p_values = np.full(np.unique(lda_df['time_ax']).shape[0], np.nan)
    for k, t in enumerate(lda_df['time_ax'].unique()):
        data_real = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 2) & (lda_df['time_ax'] == t), 'lda_align_bl']
        data_shuf = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 2) & (lda_df['time_ax'] == t), 'lda_align_bl_shuf']
        _, p_values[k] = ttest_rel(data_real, data_shuf)

    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 2)], x='time_ax', y='lda_align_bl_shuf',
                 errorbar='se', ax=axs[1, r], err_kws={'lw': 0}, zorder=0, color='grey')
    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 2)], x='time_ax', y='lda_align_bl',
                 errorbar='se', ax=axs[1, r], err_kws={'lw': 0}, zorder=1, color=colors[region])
    add_significance(np.unique(lda_df['time_ax']), p_values, ax=axs[1, r], color=colors[region])
    axs[1, r].set(ylabel='', xlabel='', xlim=[-1, 1])

f.supylabel('LDA alignment')

sns.despine(trim=True)
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.85, wspace=0.5)
plt.show()
