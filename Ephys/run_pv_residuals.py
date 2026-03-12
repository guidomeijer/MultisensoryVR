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
MIN_NEURONS = 2

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position_0mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Load recording info
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv')
rec['date'] = rec['date'].astype(str)

# Loop over recordings
pv_corr_df = pd.DataFrame()
for i in range(len(spike_dict['residuals'])):

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

        # Check if enough neurons
        if spike_counts.shape[1] < MIN_NEURONS:
            continue

        # Get unique spatial bins
        rel_pos_bins = np.unique(spatial_bins)
        n_bins = len(rel_pos_bins)

        # Calculate mean population vector per spatial bin for both contexts
        pv_mean_1 = np.zeros((n_bins, spike_counts.shape[1]))
        pv_mean_2 = np.zeros((n_bins, spike_counts.shape[1]))

        for b, bin_idx in enumerate(rel_pos_bins):
            # Context 1
            bin_mask = (spatial_bins == bin_idx) & (context_per_bin == 1)
            if np.sum(bin_mask) > 0:
                pv_mean_1[b, :] = np.mean(spike_counts[bin_mask, :], axis=0)

            # Context 2
            bin_mask = (spatial_bins == bin_idx) & (context_per_bin == 2)
            if np.sum(bin_mask) > 0:
                pv_mean_2[b, :] = np.mean(spike_counts[bin_mask, :], axis=0)

        # Correlate population vectors between the two contexts
        pv_corr = np.corrcoef(np.vstack((pv_mean_1, pv_mean_2)))
        pv_corr = pv_corr[:n_bins, n_bins:]

        # Add to dataframe
        pv_corr_df = pd.concat([pv_corr_df, pd.DataFrame([{
            'subject': rec.loc[i, 'subject'],
            'date': rec.loc[i, 'date'],
            'probe': rec.loc[i, 'probe'],
            'region': region,
            'pv_corr': pv_corr,
            'n_neurons': spike_counts.shape[1]
        }])], ignore_index=True)

# %% Plot
regions = pv_corr_df['region'].unique()
f, ax = plt.subplots(1, len(regions), figsize=(2 * len(regions), 2), dpi=dpi, squeeze=False)
ax = ax.flatten()

for i, region in enumerate(regions):
    region_df = pv_corr_df[(pv_corr_df['region'] == region)
                            & np.isin(pv_corr_df['subject'].values,
                                      subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
    pv_corr = np.mean(np.stack(region_df['pv_corr'].values), axis=0)
    sns.heatmap(pv_corr, cmap='coolwarm', vmin=-0.8, vmax=0.8, square=True, ax=ax[i], cbar=False)
    ax[i].plot([45, 45], [0, 145])
    ax[i].set(title=f'{region} (n={region_df.shape[0]})')

plt.tight_layout()
plt.savefig(path_dict['fig_path'] / 'pv_correlation_residuals_far.pdf')
plt.show()

f, ax = plt.subplots(1, len(regions), figsize=(2 * len(regions), 2), dpi=dpi, squeeze=False)
ax = ax.flatten()

for i, region in enumerate(regions):
    region_df = pv_corr_df[(pv_corr_df['region'] == region)
                            & np.isin(pv_corr_df['subject'].values,
                                      subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int))]
    pv_corr = np.mean(np.stack(region_df['pv_corr'].values), axis=0)
    sns.heatmap(pv_corr, cmap='coolwarm', vmin=-0.8, vmax=0.8, square=True, ax=ax[i], cbar=False)
    ax[i].set(title=f'{region} (n={region_df.shape[0]})',
              xticks=[0, pv_corr.shape[0]], yticks=[0, pv_corr.shape[0]],
              xticklabels=[1, pv_corr.shape[0]], yticklabels=[1, pv_corr.shape[0]],
              xlabel='Spatial bin', ylabel='Spatial bin')

plt.tight_layout()
plt.savefig(path_dict['fig_path'] / 'pv_correlation_residuals_near.pdf')
plt.show()

