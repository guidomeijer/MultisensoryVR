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

# Load in data
path_dict = paths()
subjects = load_subjects()
context_df = pd.read_csv(join(path_dict['save_path'], 'decode_context_cortex_binpairs_25neurons_regression.csv'))
context_df['region'] = context_df['region'].astype(str)


# %%

f, ax1 = plt.subplots(figsize=(3, 2.5), dpi=dpi, sharey=True)

# Create spatial bin matrix train_position by test_position and average over dates
pivot_df = context_df[context_df['region'] == 'Cortex'].groupby(['train_position', 'test_position']).mean(
    numeric_only=True)['accuracy'].unstack()

sns.heatmap(pivot_df, cmap='viridis', ax=ax1, cbar_kws={'label': 'Accuracy'})
ax1.set(xlabel='Test position (cm)', ylabel='Train position (cm)', xticks=[], yticks=[], title='Cortex')

plt.tight_layout()
plt.show()

f, ax1 = plt.subplots(figsize=(3, 2.5), dpi=dpi, sharey=True)

# Create spatial bin matrix train_position by test_position and average over dates
pivot_df = context_df[context_df['region'] == 'CA1'].groupby(['train_position', 'test_position']).mean(
    numeric_only=True)['accuracy'].unstack()

sns.heatmap(pivot_df, cmap='viridis', ax=ax1, cbar_kws={'label': 'Accuracy'})
ax1.set(xlabel='Test position (cm)', ylabel='Train position (cm)', xticks=[], yticks=[], title='CA1')

plt.tight_layout()
plt.show()