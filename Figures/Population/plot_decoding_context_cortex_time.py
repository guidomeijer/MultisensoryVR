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
context_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_subsampled_cortex_time.csv'))
context_df['region'] = context_df['region'].astype(str)

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.75 * 2, 1.75), dpi=dpi, sharey=True)
sns.lineplot(context_df[context_df['object'] == 1], x='position', y='accuracy', hue='region', errorbar='se',
             ax=ax1, err_kws={'lw': 0}, legend=None, zorder=2,
             hue_order=['Cortex', 'CA1'], palette=[colors['PERI'], colors['CA1']])


sns.lineplot(context_df[context_df['object'] == 2], x='position', y='accuracy', hue='region', errorbar='se',
             ax=ax2, err_kws={'lw': 0}, legend=None, zorder=2,
             hue_order=['Cortex', 'CA1'], palette=[colors['PERI'], colors['CA1']])

plt.show()