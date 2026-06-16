# -*- coding: utf-8 -*-
"""
Plot the results of decode_context_binpairs_per_region.py

By Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style, load_subjects
colors, dpi = figure_style()

# Settings
PLOT_REGIONS = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']

# Load in data
path_dict = paths()
file_path = join(path_dict['save_path'], 'decode_context_binpairs_randomforest.csv')
try:
    context_df = pd.read_csv(file_path, dtype={'subject': str})
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Load subjects and merge to identify Far/Near
subjects = load_subjects()
subjects['SubjectID'] = subjects['SubjectID'].astype(str)
context_df = context_df.merge(subjects[['SubjectID', 'Far']], left_on='subject', right_on='SubjectID', how='inner')
context_df = context_df[context_df['Far'].isin([0, 1])]

# %% Far objects heatmaps
plot_df = context_df[context_df['Far'] == 1]
f, axs = plt.subplots(1, 6, figsize=(7, 1.3), dpi=dpi, sharey=True)
for i, region in enumerate(PLOT_REGIONS):
    # Create spatial bin matrix train_position by test_position and average over dates
    pivot_df = plot_df[plot_df['region'] == region].groupby(['train_position', 'test_position']).mean(
        numeric_only=True)['accuracy'].unstack()
    
    sns.heatmap(pivot_df, cmap='coolwarm', ax=axs[i], vmin=0.3, vmax=0.7,
                cbar=True if i == 5 else False,
                cbar_kws={'label': 'Decoding accuracy' if i == 5 else ''})
    axs[i].plot([40, 40], [0, 150], color='w', ls='--', lw=0.75)
    axs[i].plot([0, 150], [40, 40], color='w', ls='--', lw=0.75)
    axs[i].plot([130, 130], [0, 150], color='w', ls='--', lw=0.75)
    axs[i].plot([0, 150], [130, 130], color='w', ls='--', lw=0.75)
    axs[i].set(xlabel='', ylabel='', title=region, xticks=[], yticks=[])
    axs[i].invert_yaxis()
axs[0].set(ylabel='Train position')
f.text(0.5, 0.04, 'Test position', ha='center')

plt.subplots_adjust(left=0.1, bottom=0.14, right=0.98, top=0.95, wspace=0.05, hspace=0.2)
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'decode_context_binpairs_far.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'decode_context_binpairs_far.jpg', dpi=600)
plt.show()

# %% Near objects heatmaps
plot_df = context_df[context_df['Far'] == 0]
f, axs = plt.subplots(1, 6, figsize=(7, 1.4), dpi=dpi, sharey=True)
for i, region in enumerate(PLOT_REGIONS):
    # Create spatial bin matrix train_position by test_position and average over dates
    pivot_df = plot_df[plot_df['region'] == region].groupby(['train_position', 'test_position']).mean(
        numeric_only=True)['accuracy'].unstack()
    
    sns.heatmap(pivot_df, cmap='coolwarm', ax=axs[i], vmin=0.3, vmax=0.7,
                cbar=True if i == 5 else False,
                cbar_kws={'label': 'Decoding accuracy' if i == 5 else ''})
    axs[i].plot([40, 40], [0, 150], color='w', ls='--', lw=0.75)
    axs[i].plot([0, 150], [40, 40], color='w', ls='--', lw=0.75)
    axs[i].plot([85, 85], [0, 150], color='w', ls='--', lw=0.75)
    axs[i].plot([0, 150], [85, 85], color='w', ls='--', lw=0.75)
    axs[i].set(xlabel='', ylabel='', title=region, xticks=[], yticks=[])
    axs[i].invert_yaxis()
axs[0].set(ylabel='Train position')
f.text(0.5, 0.04, 'Test position', ha='center')

plt.subplots_adjust(left=0.1, bottom=0.14, right=0.98, top=0.95, wspace=0.05, hspace=0.2)
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'decode_context_binpairs_near.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'decode_context_binpairs_near.jpg', dpi=600)
plt.show()

# %% 

f, axs = plt.subplots(1, 6, figsize=(7, 1.5), dpi=dpi, sharey=True)
for i, region in enumerate(PLOT_REGIONS):
    region_df = context_df[context_df['region'] == region]
    
    # Far: tested at second landmark 130 - 150 cm (1300 - 1500 mm)
    far_df = region_df[(region_df['Far'] == 1) & (region_df['test_position'] >= 1300) & (region_df['test_position'] <= 1500)]
    plot_df = far_df.groupby(['subject', 'date', 'train_position']).mean(numeric_only=True).reset_index()
    plot_df['train_position_cm'] = plot_df['train_position'] / 10
    
    sns.lineplot(data=plot_df, x='train_position_cm', y='accuracy', errorbar='se', ax=axs[i],
                 err_kws={'lw': 0}, color='k')
    axs[i].axhline(0.5, ls='--', color='k', lw=0.5)
    axs[i].set(xlabel='', ylabel='', ylim=[0.3, 0.74], xticks=[0, 50, 100, 150],
               yticks=[0.3, 0.4, 0.5, 0.6, 0.7])
    axs[i].text(15, 0.65, region, color=colors[region], weight='bold')
sns.despine(trim=True)

axs[0].set(ylabel='Decoding accuracy')
f.text(0.5, 0.04, 'Train position (cm)', ha='center')
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.98, top=1, wspace=0.15)
plt.show()
