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
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style, add_significance
colors, dpi = figure_style()

# Load in data
path_dict = paths()
subjects = load_subjects()
cortex_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_subsampled_cortex_time.csv'))
cortex_df['region'] = cortex_df['region'].astype(str)
regions_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_subsampled_time.csv'))
regions_df['region'] = regions_df['region'].astype(str)

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.75 * 2, 1.75), dpi=dpi, sharey=True)
ax1.plot([0, 0], [0.4, 0.7], color='grey', ls='--')
sns.lineplot(cortex_df[cortex_df['object'] == 1], x='time', y='accuracy', hue='region', errorbar='se',
             ax=ax1, err_kws={'lw': 0}, legend=None, zorder=2,
             hue_order=['Cortex', 'CA1'], palette=[colors['PERI'], colors['CA1']])
ax1.set(yticks=[0.4, 0.5, 0.6, 0.7], yticklabels=[40, 50, 60, 70], xticks=[-2, -1, 0, 1, 2],
        ylabel='Context decoding accuracy (%)', xlabel='')

ax2.plot([0, 0], [0.4, 0.7], color='grey', ls='--')
sns.lineplot(cortex_df[cortex_df['object'] == 2], x='time', y='accuracy', hue='region', errorbar='se',
             ax=ax2, err_kws={'lw': 0}, legend=None, zorder=2,
             hue_order=['Cortex', 'CA1'], palette=[colors['PERI'], colors['CA1']])
ax2.set(yticks=[0.4, 0.5, 0.6, 0.7], xlabel='', xticks=[-2, -1, 0, 1, 2])

f.supxlabel('Time from object entry (s)', fontsize=7, y=0.08)
sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %%

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 1.75), dpi=dpi, sharey=True)

ax1.plot([0, 0], [0.4, 0.7], color='grey', ls='--')
sns.lineplot(regions_df[regions_df['object'] == 1], x='time', y='accuracy', hue='region', errorbar='se',
             ax=ax1, err_kws={'lw': 0}, legend='brief', zorder=2, palette=colors)
ax1.set(yticks=[0.4, 0.5, 0.6, 0.7], yticklabels=[40, 50, 60, 70], xticks=[-2, -1, 0, 1, 2],
        ylabel='Context decoding accuracy (%)', xlim=[-0.2, 1], xlabel='Time from object entry (s)')
ax1.legend(bbox_to_anchor=(1.1, 0.4))
sns.despine(trim=True)
plt.tight_layout()
plt.show()