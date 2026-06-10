# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 09/06/2026
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
MIN_NEURONS = 10
FIRST_OBJ = 450
BRIDGE_FAR = 900
FAR_OBJ = 1350
BRIDGE_NEAR = 700
NEAR_OBJ = 900
AFTER_OBJ = 70

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

    # Get subject
    subject = spike_dict['subject'][i]
    is_far = subjects.loc[subjects['SubjectID'] == subject, 'Far'].values[0]

    # Get brain regions
    regions = spike_dict['region'][i]
    if regions is None:
        continue
    unique_regions = np.unique(regions)

    # Loop over brain regions
    for r, region in enumerate(unique_regions):
        if region == 'root':
            continue

        # Select neurons from this brain region
        region_mask = regions == region
        spike_counts = spike_dict['residuals'][i][:, region_mask]  # spatial bins x neurons
        spatial_bins = spike_dict['position'][i]
        context_per_bin = spike_dict['context'][i]

        # Get average population vector per condition
        pop_vec = dict()
        pop_vec['objA_cond1'] = np.mean(spike_counts[(spatial_bins == FIRST_OBJ + AFTER_OBJ) & (context_per_bin == 1), :], axis=0)
        pop_vec['objA_cond2'] = np.mean(spike_counts[(spatial_bins == FIRST_OBJ + AFTER_OBJ) & (context_per_bin == 2), :], axis=0)
        if is_far:
            pop_vec['objB_cond1'] = np.mean(spike_counts[(spatial_bins == FAR_OBJ + AFTER_OBJ) & (context_per_bin == 1), :], axis=0)
            pop_vec['objB_cond2'] = np.mean(spike_counts[(spatial_bins == FAR_OBJ + AFTER_OBJ) & (context_per_bin == 2), :], axis=0)
            pop_vec['bridge_cond1'] = np.mean(spike_counts[(spatial_bins == BRIDGE_FAR) & (context_per_bin == 1), :], axis=0)
            pop_vec['bridge_cond2'] = np.mean(spike_counts[(spatial_bins == BRIDGE_FAR) & (context_per_bin == 2), :], axis=0)
        else:
            pop_vec['objB_cond1'] = np.mean(spike_counts[(spatial_bins == NEAR_OBJ + AFTER_OBJ) & (context_per_bin == 1), :], axis=0)
            pop_vec['objB_cond2'] = np.mean(spike_counts[(spatial_bins == NEAR_OBJ + AFTER_OBJ) & (context_per_bin == 2), :], axis=0)
            pop_vec['bridge_cond1'] = np.mean(spike_counts[(spatial_bins == BRIDGE_NEAR) & (context_per_bin == 1), :], axis=0)
            pop_vec['bridge_cond2'] = np.mean(spike_counts[(spatial_bins == BRIDGE_NEAR) & (context_per_bin == 2), :], axis=0)



        # Get unique spatial bins
        rel_pos_bins = np.unique(spatial_bins)
        n_bins = len(rel_pos_bins)

        # Check if enough neurons
        if spike_counts.shape[1] < MIN_NEURONS:
            continue

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

        # Calculate PV angle in degrees
        dot_products = np.dot(pv_mean_1, pv_mean_2.T)
        norms_1 = np.linalg.norm(pv_mean_1, axis=1)
        norms_2 = np.linalg.norm(pv_mean_2, axis=1)
        cosine_sim = dot_products / np.outer(norms_1, norms_2)
        pv_angle = np.degrees(np.arccos(np.clip(cosine_sim, -1.0, 1.0)))