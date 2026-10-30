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
regions_df = pd.read_csv(join(path_dict['save_path'], 'decode_object_time.csv'))
regions_df['region'] = regions_df['region'].astype(str)

# %%

f, ax1 = plt.subplots(1, 1, figsize=(2.2, 1.75), dpi=dpi, sharey=True)

ax1.plot([-1, 1], [0.5, 0.5], color='grey', ls='--')
sns.lineplot(regions_df, x='time', y='accuracy', hue='region', errorbar='se',
             ax=ax1, err_kws={'lw': 0}, legend='brief', zorder=2, palette=colors)
ax1.set(yticks=[0.5, 0.6, 0.7, 0.8, 0.9], yticklabels=[50, 60, 70, 80, 90], ylim=[0.45, 0.9],
        xticks=[-1, 0, 1], ylabel='Accuracy (%)', xlim=[-1, 1], title='Object decoding',
        xlabel='Time from object entry (s)')
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'object_decoding.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'object_decoding.pdf')
plt.show()