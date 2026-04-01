# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2024

By Guido Meijer
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter1d
from msvr_functions import paths, load_subjects

# Initialize
subjects = load_subjects()
path_dict = paths(sync=False)

# Settings
FAR = 1
SMOOTH_SIGMA = 0.5  # Standard deviation for Gaussian spatial smoothing

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position_0mms.pickle', 'rb') as handle:
    residuals_dict = pickle.load(handle)

# Change all cortex regions to 'Cortex'
targets = {'VIS', 'AUD', 'TEa', 'PERI', 'LEC'}
residuals_dict['region'] = [
    ['Cortex' if item in targets else item for item in sublist]
    for sublist in residuals_dict['region']]
residuals_dict['region'] = [np.array(i) for i in residuals_dict['region']]

# Get unique spatial bins across the whole dataset
all_bins = np.unique(np.concatenate([np.unique(p).astype(int) for p in residuals_dict['position']]))
all_bins.sort()

# Set up the plotting (2 rows: Cortex and CA1; 2 columns: Trajectories and Distance)
fig = plt.figure(figsize=(15, 12))
regions = ['Cortex', 'CA1']

for r_idx, region in enumerate(regions):
    mean_res_c1 = []
    mean_res_c2 = []

    # Iterate over sessions to extract mean neural activity per bin and context for this region
    for i in range(len(residuals_dict['residuals'])):

        # Skip if not the right subject
        if residuals_dict['subject'][i] in subjects.loc[subjects['Far'] == FAR, 'SubjectID'].values:

            # Filter neurons for the specific region
            region_mask = residuals_dict['region'][i] == region
            if not np.any(region_mask):
                continue

            res_session = residuals_dict['residuals'][i][:, region_mask]
            pos_session = residuals_dict['position'][i].astype(int)
            ctx_session = residuals_dict['context'][i]

            # Initialize session mean matrices: [bins x neurons]
            n_neurons = res_session.shape[1]
            session_c1 = np.full((len(all_bins), n_neurons), np.nan)
            session_c2 = np.full((len(all_bins), n_neurons), np.nan)

            for b_idx, bin_val in enumerate(all_bins):
                # Mask for this bin and context 1
                mask_c1 = (pos_session == bin_val) & (ctx_session == 1)
                if np.any(mask_c1):
                    session_c1[b_idx, :] = np.mean(res_session[mask_c1, :], axis=0)

                # Mask for this bin and context 2
                mask_c2 = (pos_session == bin_val) & (ctx_session == 2)
                if np.any(mask_c2):
                    session_c2[b_idx, :] = np.mean(res_session[mask_c2, :], axis=0)

            mean_res_c1.append(session_c1)
            mean_res_c2.append(session_c2)

    # Concatenate all neurons from all sessions horizontally
    X_c1 = np.hstack(mean_res_c1)
    X_c2 = np.hstack(mean_res_c2)

    # Only keep bins that have data in both contexts across all sessions (no NaNs)
    valid_bins_mask = ~np.any(np.isnan(X_c1), axis=1) & ~np.any(np.isnan(X_c2), axis=1)
    X_c1_clean = X_c1[valid_bins_mask, :]
    X_c2_clean = X_c2[valid_bins_mask, :]
    clean_bins = all_bins[valid_bins_mask]

    # Apply spatial smoothing
    X_c1_clean = gaussian_filter1d(X_c1_clean, sigma=SMOOTH_SIGMA, axis=0)
    X_c2_clean = gaussian_filter1d(X_c2_clean, sigma=SMOOTH_SIGMA, axis=0)

    # Concatenate both contexts to fit a shared PCA space
    X_combined = np.vstack([X_c1_clean, X_c2_clean])

    # Standardize features (z-score neurons)
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)

    pca = PCA(n_components=10)
    pca.fit(X_combined_scaled)

    # Project back to get trajectories
    proj_c1 = pca.transform(scaler.transform(X_c1_clean))
    proj_c2 = pca.transform(scaler.transform(X_c2_clean))

    # Calculate Euclidean distance between trajectories in PCA space (using all components)
    dist = [euclidean(proj_c1[b, :], proj_c2[b, :]) for b in range(proj_c1.shape[0])]

    # --- Plotting for this region ---
    ax_traj = fig.add_subplot(2, 2, r_idx*2 + 1, projection='3d')
    ax_dist = fig.add_subplot(2, 2, r_idx*2 + 2)

    # 1. PCA Trajectories (First 3 PCs)
    ax_traj.plot(proj_c1[:, 0], proj_c1[:, 1], proj_c1[:, 2], '-o', label='Context 1', markersize=4)
    ax_traj.plot(proj_c2[:, 0], proj_c2[:, 1], proj_c2[:, 2], '-o', label='Context 2', markersize=4)
    # Mark starting point
    ax_traj.scatter(proj_c1[0, 0], proj_c1[0, 1], proj_c1[0, 2], color='green', marker='x', s=50, zorder=5, label='Start')
    ax_traj.scatter(proj_c2[0, 0], proj_c2[0, 1], proj_c2[0, 2], color='green', marker='x', s=50, zorder=5)

    ax_traj.set_xlabel('PC 1')
    ax_traj.set_ylabel('PC 2')
    ax_traj.set_zlabel('PC 3')
    ax_traj.set_title(f'{region}: Trajectories')
    ax_traj.legend()

    # 2. Euclidean Distance over Position
    ax_dist.plot(clean_bins, dist, color='black', linewidth=2)
    ax_dist.set_xlabel('Position Bin')
    ax_dist.set_ylabel('Euclidean Distance')
    ax_dist.set_title(f'{region}: Divergence')

    # Optional: Save the divergence data for each region
    dist_df = pd.DataFrame({
        'position': clean_bins,
        'euclidean_dist': dist
    })
    dist_df.to_csv(path_dict['save_path'] / f'pca_divergence_{region}.csv', index=False)

plt.tight_layout()
plt.show()
