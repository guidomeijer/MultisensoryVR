# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 12/03/2026
"""
# %%

import numpy as np
import pandas as pd
import pickle
from msvr_functions import paths

# Settings
MIN_NEURONS = 2

# Initialize
path_dict = paths(sync=False)

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position.pickle', 'rb') as handle:
    residuals_dict = pickle.load(handle)

# Load recording info
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv')
rec['date'] = rec['date'].astype(str)

# Loop over recordings
decode_df = pd.DataFrame()
for i in range(len(residuals_dict['residuals'])):
    print(f'Recording {i} of {len(residuals_dict["residuals"])}')

    # Get brain regions
    regions = residuals_dict['region'][i]
    if regions is None:
        continue

    unique_regions = np.unique(regions)

    # Decode per brain region
    for r, region in enumerate(unique_regions):
        if region == 'root':
            continue
        print(f'Processing {region}')

        # Select neurons from this brain region
        region_mask = regions == region
        spike_counts = residuals_dict['residuals'][i][:, region_mask]  # spatial bins x neurons
        spatial_bins = residuals_dict['position'][i]
        context_per_bin = residuals_dict['context'][i]
        
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
        decode_df = pd.concat([decode_df, pd.DataFrame([{
            'subject': rec.loc[i, 'subject'],
            'date': rec.loc[i, 'date'],
            'probe': rec.loc[i, 'probe'],
            'region': region,
            'pv_corr': pv_corr,
            'n_neurons': spike_counts.shape[1]
        }])], ignore_index=True)

# Save dataframe
save_path = path_dict['google_drive_data_path'] / 'pv_correlation.pickle'
decode_df.to_pickle(save_path)
print(f'Saved results to {save_path}')
