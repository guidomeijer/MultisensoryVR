# -*- coding: utf-8 -*-
"""
Plot the results of decode_context_binpairs_per_region.py

By Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import mne
import matplotlib.pyplot as plt
from scipy import stats
from msvr_functions import paths, figure_style, load_subjects, add_significance
mne.set_log_level('WARNING')
colors, dpi = figure_style()

# Settings
PLOT_REGIONS = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']

# Load in data
path_dict = paths()
file_path = join(path_dict['google_drive_data_path'], 'decode_context_binpairs_randomforest.csv')
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
LINE_COLOR = 'grey'

plot_df = context_df[context_df['Far'] == 1]
f, axs = plt.subplots(1, 6, figsize=(7, 1.5), dpi=dpi, sharey=True)
for i, region in enumerate(PLOT_REGIONS):
    # Create spatial bin matrix train_position by test_position and average over dates
    pivot_df = plot_df[plot_df['region'] == region].groupby(['train_position', 'test_position']).mean(
        numeric_only=True)['accuracy'].unstack()
    
    sns.heatmap(pivot_df, cmap='RdBu_r', ax=axs[i], vmin=0.3, vmax=0.7, cbar=False)
    axs[i].plot([40, 40], [0, 150], color=LINE_COLOR, ls='--', lw=0.5)
    axs[i].plot([0, 150], [40, 40], color=LINE_COLOR, ls='--', lw=0.5)
    axs[i].plot([130, 130], [0, 150], color=LINE_COLOR, ls='--', lw=0.5)
    axs[i].plot([0, 150], [130, 130], color=LINE_COLOR, ls='--', lw=0.5)
    axs[i].set(xlabel='', ylabel='', title=region, xticks=[], yticks=[])
    axs[i].invert_yaxis()
axs[0].set(ylabel='Train position')
f.text(0.5, 0.04, 'Test position', ha='center')

# Add a single colorbar in the right margin to keep all subplot widths perfectly equal
# Coordinates: [left, bottom, width, height] relative to the figure
cbar_ax = f.add_axes([0.93, 0.14, 0.01, 0.71])
cbar = f.colorbar(axs[5].collections[0], cax=cbar_ax)
cbar.set_label('Decoding accuracy', fontsize=7, rotation=270, labelpad=10)

plt.subplots_adjust(left=0.04, bottom=0.14, right=0.92, top=0.85, wspace=0.05, hspace=0.2)
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
    
    sns.heatmap(pivot_df, cmap='RdBu_r', ax=axs[i], vmin=0.3, vmax=0.7, cbar=False)
    axs[i].plot([40, 40], [0, 150], color=LINE_COLOR, ls='--', lw=0.5)
    axs[i].plot([0, 150], [40, 40], color=LINE_COLOR, ls='--', lw=0.5)
    axs[i].plot([85, 85], [0, 150], color=LINE_COLOR, ls='--', lw=0.5)
    axs[i].plot([0, 150], [85, 85], color=LINE_COLOR, ls='--', lw=0.5)
    axs[i].set(xlabel='', ylabel='', title=region, xticks=[], yticks=[])
    axs[i].invert_yaxis()
axs[0].set(ylabel='Train position')
f.text(0.5, 0.04, 'Test position', ha='center')

cbar_ax = f.add_axes([0.93, 0.14, 0.01, 0.71])
cbar = f.colorbar(axs[5].collections[0], cax=cbar_ax)
cbar.set_label('Decoding accuracy')

plt.subplots_adjust(left=0.1, bottom=0.14, right=0.90, top=0.95, wspace=0.05, hspace=0.2)
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'decode_context_binpairs_near.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'decode_context_binpairs_near.jpg', dpi=600)
plt.show()

# %% Plot second landmark decoding


f, axs = plt.subplots(1, 6, figsize=(7, 1.5), dpi=dpi, sharey=True)
p_values = {} # Dictionary to store p-values for each region
for i, region in enumerate(PLOT_REGIONS):
    region_df = context_df[context_df['region'] == region]
    
    # Far: tested at second landmark 130 - 150 cm (1300 - 1500 mm)
    far_df = region_df[(region_df['Far'] == 1) & (region_df['test_position'] >= 1350) & (region_df['test_position'] <= 1450)]
    plot_df = far_df.groupby(['subject', 'date', 'train_position']).mean(numeric_only=True).reset_index()
    plot_df['train_position_cm'] = plot_df['train_position'] / 10

    # Control: tested before first landmark: 0 - 20 cm
    control_df = region_df[(region_df['Far'] == 1) & (region_df['test_position'] >= 0) & (region_df['test_position'] <= 200)]
    control_df = control_df.groupby(['subject', 'date', 'train_position']).mean(numeric_only=True).reset_index()
    control_df['train_position_cm'] = control_df['train_position'] / 10

    # Do statistics
    test_matrix = plot_df.pivot(index=['subject', 'date'], columns='train_position_cm', values='accuracy')
    control_matrix = control_df.pivot(index=['subject', 'date'], columns='train_position_cm', values='accuracy')
    positions = test_matrix.columns.values
    X = test_matrix.values - control_matrix.values
    t_threshold = stats.t.ppf(1 - 0.2 / 2, test_matrix.shape[0]-1)
    t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
            X,
            threshold=t_threshold,
            n_permutations=1000, 
            tail=0,          # Two-tailed test
            out_type='mask'  # Returns boolean masks for positions
        )
    region_p_values = np.ones(len(positions))
    for cluster_mask, p_val in zip(clusters, cluster_p_values):
        if p_val < 0.05:
            # Assign the cluster-level p-value to all positions in this cluster
            region_p_values[cluster_mask] = p_val

    sns.lineplot(data=plot_df, x='train_position_cm', y='accuracy', errorbar='se', ax=axs[i],
                 err_kws={'lw': 0}, color='k')
    add_significance(np.unique(plot_df['train_position_cm']), region_p_values, ax=axs[i], y_pos=0.78)
    axs[i].axhline(0.5, ls='--', color='k', lw=0.5)
    axs[i].set(xlabel='', ylabel='', ylim=[0.3, 0.8], xticks=[0, 50, 100, 150],
               yticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8], yticklabels=[30, 40, 50, 60, 70, 80])
    axs[i].text(15, 0.65, region, color=colors[region], weight='bold') # Original region text

sns.despine(trim=True)

axs[0].set(ylabel='Decoding accuracy (%)')
f.text(0.5, 0.04, 'Train position (cm)', ha='center')
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.98, top=0.95, wspace=0.15)
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'decode_context_second_landmark.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'Decoding' / 'decode_context_second_landmark.jpg', dpi=600) # Original save
plt.show()
