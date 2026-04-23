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
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style, add_significance
colors, dpi = figure_style()
path_dict = paths()

# Settings
REGIONS = ['CA1', 'LEC', 'PERI', 'TEa', 'AUD', 'VIS']

# Load in data
lda_df = pd.read_csv(path_dict['google_drive_data_path'] / 'lda_alignment_cosim_0.6_-0.15.csv')
lda_df = lda_df[(lda_df['time_ax'] >= -1) & (lda_df['time_ax'] <= 1)]

# %% Plot

f, axs = plt.subplots(1, 6, figsize=(5.25, 1.75), dpi=dpi, sharey=True)
for r, region in enumerate(REGIONS):

    # Perform t-test per timepoint
    p_values = np.full(np.unique(lda_df['time_ax']).shape[0], np.nan)
    for k, t in enumerate(lda_df['time_ax'].unique()):
        data_real = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 2) & (lda_df['time_ax'] == t), 'lda_align_bl']
        data_shuf = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 2) & (lda_df['time_ax'] == t), 'lda_align_bl_shuf']
        _, p_values[k] = ttest_rel(data_real, data_shuf)

    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 2)], x='time_ax', y='lda_align_bl_shuf',
                 errorbar='se', ax=axs[r], err_kws={'lw': 0}, zorder=0, color='grey')
    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 2)], x='time_ax', y='lda_align_bl',
                 errorbar='se', ax=axs[r], err_kws={'lw': 0}, zorder=1, color=colors[region])
    add_significance(np.unique(lda_df['time_ax']), p_values, ax=axs[r], color=colors[region], y_pos=0.1)
    axs[r].set(ylabel='', xlabel='', xlim=[-1, 1], ylim=[-0.3, 0.1], yticks=[-0.3, -0.2, -0.1, 0, 0.1],
               yticklabels=[-0.3, -0.2, -0.1, 0, 0.1], title=region)
axs[0].set(ylabel='Alignment to LDA axis\n(cosine similarity)')
f.supxlabel('Time from ripple onset (s)', y=0.05, fontsize=7)

sns.despine(trim=True)
plt.subplots_adjust(left=0.11, bottom=0.25, right=0.98, top=0.85, wspace=0.3)
#plt.tight_layout(w_pad=0.5)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'LDA_alignment_obj2_bl.pdf')
plt.show()

# %%
f, axs = plt.subplots(1, 6, figsize=(5.25, 1.75), dpi=dpi, sharey=True)
for r, region in enumerate(REGIONS):

    # Perform t-test per timepoint
    p_values = np.full(np.unique(lda_df['time_ax']).shape[0], np.nan)
    for k, t in enumerate(lda_df['time_ax'].unique()):
        data_real = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 2) & (lda_df['time_ax'] == t), 'lda_align']
        data_shuf = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 2) & (lda_df['time_ax'] == t), 'lda_align_shuf']
        _, p_values[k] = ttest_rel(data_real, data_shuf)

    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 2)], x='time_ax', y='lda_align_shuf',
                 errorbar='se', ax=axs[r], err_kws={'lw': 0}, zorder=0, color='grey')
    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 2)], x='time_ax', y='lda_align',
                 errorbar='se', ax=axs[r], err_kws={'lw': 0}, zorder=1, color=colors[region])
    add_significance(np.unique(lda_df['time_ax']), p_values, ax=axs[r], color=colors[region], y_pos=0.1)
    axs[r].set(ylabel='', xlabel='', xlim=[-1, 1], ylim=[-0.3, 0.1], yticks=[-0.3, -0.2, -0.1, 0, 0.1],
               yticklabels=[-0.3, -0.2, -0.1, 0, 0.1], title=region)
axs[0].set(ylabel='Alignment to LDA axis\n(cosine similarity)')
f.supxlabel('Time from ripple onset (s)', y=0.05, fontsize=7)

sns.despine(trim=True)
plt.subplots_adjust(left=0.11, bottom=0.25, right=0.98, top=0.85, wspace=0.3)
#plt.tight_layout(w_pad=0.5)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'LDA_alignment_obj2.pdf')
plt.show()

# %%

f, axs = plt.subplots(1, 6, figsize=(5.25, 1.75), dpi=dpi, sharey=True)
for r, region in enumerate(REGIONS):

    # Perform t-test per timepoint
    p_values = np.full(np.unique(lda_df['time_ax']).shape[0], np.nan)
    for k, t in enumerate(lda_df['time_ax'].unique()):
        data_real = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 1) & (lda_df['time_ax'] == t), 'lda_align_bl']
        data_shuf = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 1) & (lda_df['time_ax'] == t), 'lda_align_bl_shuf']
        _, p_values[k] = ttest_rel(data_real, data_shuf)

    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 1)], x='time_ax', y='lda_align_bl_shuf',
                 errorbar='se', ax=axs[r], err_kws={'lw': 0}, zorder=0, color='grey')
    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 1)], x='time_ax', y='lda_align_bl',
                 errorbar='se', ax=axs[r], err_kws={'lw': 0}, zorder=1, color=colors[region])
    add_significance(np.unique(lda_df['time_ax']), p_values, ax=axs[r], color=colors[region], y_pos=0.1)
    axs[r].set(ylabel='', xlabel='', xlim=[-1, 1], ylim=[-0.3, 0.1], yticks=[-0.3, -0.2, -0.1, 0, 0.1],
               yticklabels=[-0.3, -0.2, -0.1, 0, 0.1], title=region)
axs[0].set(ylabel='Alignment to LDA axis\n(cosine similarity)')
f.supxlabel('Time from ripple onset (s)', y=0.05, fontsize=7)

sns.despine(trim=True)
plt.subplots_adjust(left=0.11, bottom=0.25, right=0.98, top=0.85, wspace=0.3)
#plt.tight_layout(w_pad=0.5)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'LDA_alignment_obj1_bl.pdf')
plt.show()

# %%
f, axs = plt.subplots(1, 6, figsize=(5.25, 1.75), dpi=dpi, sharey=True)
for r, region in enumerate(REGIONS):

    # Perform t-test per timepoint
    p_values = np.full(np.unique(lda_df['time_ax']).shape[0], np.nan)
    for k, t in enumerate(lda_df['time_ax'].unique()):
        data_real = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 1) & (lda_df['time_ax'] == t), 'lda_align']
        data_shuf = lda_df.loc[(lda_df['region'] == region) & (lda_df['object'] == 1) & (lda_df['time_ax'] == t), 'lda_align_shuf']
        _, p_values[k] = ttest_rel(data_real, data_shuf)

    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 1)], x='time_ax', y='lda_align_shuf',
                 errorbar='se', ax=axs[r], err_kws={'lw': 0}, zorder=0, color='grey')
    sns.lineplot(data=lda_df[(lda_df['region'] == region) & (lda_df['object'] == 1)], x='time_ax', y='lda_align',
                 errorbar='se', ax=axs[r], err_kws={'lw': 0}, zorder=1, color=colors[region])
    add_significance(np.unique(lda_df['time_ax']), p_values, ax=axs[r], color=colors[region], y_pos=0.1)
    axs[r].set(ylabel='', xlabel='', xlim=[-1, 1], ylim=[-0.3, 0.1], yticks=[-0.3, -0.2, -0.1, 0, 0.1],
               yticklabels=[-0.3, -0.2, -0.1, 0, 0.1], title=region)
axs[0].set(ylabel='Alignment to LDA axis\n(cosine similarity)')
f.supxlabel('Time from ripple onset (s)', y=0.05, fontsize=7)

sns.despine(trim=True)
plt.subplots_adjust(left=0.11, bottom=0.25, right=0.98, top=0.85, wspace=0.3)
#plt.tight_layout(w_pad=0.5)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'LDA_alignment_obj1.pdf')
plt.show()