# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 12/03/2026
"""
# %%

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from msvr_functions import paths, load_subjects, figure_style
colors, dpi = figure_style()

# Settings
MIN_NEURONS = 1
FIRST_OBJ = 45
FAR_OBJ = 135
NEAR_OBJ = 90
OBJ_SIZE = 5

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position_20mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Load recording info
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv')
rec['date'] = rec['date'].astype(str)

# Get subject groups
far_subjects = subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int)
near_subjects = subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int)

# Dictionary to store pooled activity
pooled_activity = {'Far': {}, 'Near': {}}
rel_pos_bins = None

# Loop over recordings
for i in range(len(spike_dict['residuals'])):

    # Check subject group
    subject = int(rec.loc[i, 'subject'])
    if subject in far_subjects:
        group = 'Far'
    elif subject in near_subjects:
        group = 'Near'
    else:
        continue

    # Get brain regions
    regions = spike_dict['region'][i]
    if regions is None:
        continue
    unique_regions = np.unique(regions)

    # Decode per brain region
    for r, region in enumerate(unique_regions):
        if region == 'root':
            continue

        # Select neurons from this brain region
        region_mask = regions == region
        spike_counts = spike_dict['residuals'][i][:, region_mask]  # spatial bins x neurons
        spatial_bins = spike_dict['position'][i]
        context_per_bin = spike_dict['context'][i]

        # Initialize bins if not done yet
        if rel_pos_bins is None:
            rel_pos_bins = np.unique(spatial_bins)

        # Check if enough neurons
        if spike_counts.shape[1] < MIN_NEURONS:
            continue

        # Calculate mean population vector per spatial bin for both contexts
        pv_mean_1 = np.zeros((len(rel_pos_bins), spike_counts.shape[1]))
        pv_mean_2 = np.zeros((len(rel_pos_bins), spike_counts.shape[1]))

        for b, bin_idx in enumerate(rel_pos_bins):
            # Context 1
            bin_mask = (spatial_bins == bin_idx) & (context_per_bin == 1)
            if np.sum(bin_mask) > 0:
                pv_mean_1[b, :] = np.mean(spike_counts[bin_mask, :], axis=0)

            # Context 2
            bin_mask = (spatial_bins == bin_idx) & (context_per_bin == 2)
            if np.sum(bin_mask) > 0:
                pv_mean_2[b, :] = np.mean(spike_counts[bin_mask, :], axis=0)

        # Store in pooled dictionary
        if region not in pooled_activity[group]:
            pooled_activity[group][region] = {'pv1': [], 'pv2': []}
        pooled_activity[group][region]['pv1'].append(pv_mean_1)
        pooled_activity[group][region]['pv2'].append(pv_mean_2)

# %% Plot
rel_pos_bins_m = rel_pos_bins / 10

for group, filename_suffix, obj_pos in zip(['Far', 'Near'], ['far', 'near'], [FAR_OBJ, NEAR_OBJ]):
    regions = list(pooled_activity[group].keys())
    if len(regions) == 0:
        continue

    f, ax = plt.subplots(1, len(regions), figsize=(2 * len(regions), 2), dpi=dpi, squeeze=False)
    ax = ax.flatten()

    for i, region in enumerate(regions):
        # Stack neurons
        if len(pooled_activity[group][region]['pv1']) == 0:
            continue

        pv1_stack = np.hstack(pooled_activity[group][region]['pv1'])
        pv2_stack = np.hstack(pooled_activity[group][region]['pv2'])

        # Correlate
        pv_corr = np.corrcoef(np.vstack((pv1_stack, pv2_stack)))
        n_bins = pv1_stack.shape[0]
        pv_corr = pv_corr[:n_bins, n_bins:]

        # Plot
        sns.heatmap(pv_corr, cmap='coolwarm', vmin=-0.8, vmax=0.8, square=True, ax=ax[i], cbar=False)

        # Plot lines
        for pos in [FIRST_OBJ, obj_pos]:
            ax[i].plot([np.abs(rel_pos_bins_m - pos).argmin(), np.abs(rel_pos_bins_m - pos).argmin()],
                       [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
            ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - pos).argmin(), np.abs(rel_pos_bins_m - pos).argmin()],
                       color='white', linestyle='--', lw=0.5)
            ax[i].plot([np.abs(rel_pos_bins_m - (pos + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (pos + OBJ_SIZE)).argmin()],
                       [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
            ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - (pos + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (pos + OBJ_SIZE)).argmin()],
                       color='white', linestyle='--', lw=0.5)

        ax[i].set(title=f'{region} (n={pv1_stack.shape[1]})', xticks=[], yticks=[])

    plt.tight_layout()
    plt.savefig(path_dict['fig_path'] / f'pv_correlation_residuals_{filename_suffix}_pooled.pdf')
    plt.show()
