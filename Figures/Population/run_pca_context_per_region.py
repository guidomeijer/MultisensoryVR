# -*- coding: utf-8 -*-
"""
Fits PCA (10 dimensions) jointly over Context 1 and Context 2 across position bins
for each brain region so that both contexts reside in the same PC space.
Plots the population trajectories for the first two PCs and the Euclidean distance
between Context 1 and Context 2 in the shared 10D PC space over position.

By Guido Meijer
"""
# %% Imports

import pickle
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from msvr_functions import paths, load_subjects, figure_style

colors, dpi = figure_style()

# %% Settings

N_COMPONENTS = 15
MIN_NEURONS = 5
REGION_ORDER = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']
FIRST_OBJ = 45   # cm
FAR_OBJ = 135    # cm
NEAR_OBJ = 90    # cm
OBJ_SIZE = 5     # cm

# %% Load data

path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)

# Load residual spike counts over position
data_file = path_dict['google_drive_data_path'] / 'residuals_position_20mms.pickle'
if not data_file.is_file():
    data_file = path_dict['google_drive_data_path'] / 'residuals_position.pickle'

with open(data_file, 'rb') as handle:
    spike_dict = pickle.load(handle)

# %% Extract pseudo-population activity per region over position bins

# Get common spatial bins (converted to cm)
all_positions = []
for pos in spike_dict['position']:
    all_positions.extend(pos)
unique_positions_mm = np.sort(np.unique(all_positions))
unique_positions_cm = unique_positions_mm / 10.0
n_bins = len(unique_positions_mm)

# Find landmark bin indices for plotting markers
idx_start = 0
idx_first_obj = np.argmin(np.abs(unique_positions_cm - FIRST_OBJ))
idx_far_obj = np.argmin(np.abs(unique_positions_cm - FAR_OBJ))
idx_end = n_bins - 1

# Collect trial-averaged neural activity per region
region_activity_c1 = {r: [] for r in REGION_ORDER}
region_activity_c2 = {r: [] for r in REGION_ORDER}
session_pca_records = []

for i in range(len(spike_dict['residuals'])):
    this_subject = str(spike_dict['subject'][i])
    this_ses = str(spike_dict['date'][i])
    this_probe = str(spike_dict['probe'][i])
    
    # Get Far / Near session info if available
    sub_match = subjects.loc[subjects['SubjectID'].astype(str) == this_subject, 'Far']
    is_far = sub_match.values[0] if len(sub_match) > 0 else np.nan
    if is_far == 0:
        continue  # Skip Near sessions for this analysis

    spatial_bins = spike_dict['position'][i]
    context_per_bin = spike_dict['context'][i]
    residuals = spike_dict['residuals'][i]
    regions = spike_dict['region'][i]

    if regions is None or residuals is None:
        continue

    for region in REGION_ORDER:
        region_mask = (regions == region)
        if np.sum(region_mask) < MIN_NEURONS:
            continue

        spike_counts = residuals[:, region_mask]

        # Filter out neurons with zero variance
        valid_neurons = np.std(spike_counts, axis=0) > 0
        spike_counts = spike_counts[:, valid_neurons]
        if spike_counts.shape[1] < MIN_NEURONS:
            continue

        # Compute mean activity per position bin for Context 1 and Context 2
        mean_c1 = np.full((n_bins, spike_counts.shape[1]), np.nan)
        mean_c2 = np.full((n_bins, spike_counts.shape[1]), np.nan)

        for b, pos in enumerate(unique_positions_mm):
            mask_1 = (spatial_bins == pos) & (context_per_bin == 1)
            mask_2 = (spatial_bins == pos) & (context_per_bin == 2)

            if np.sum(mask_1) > 0:
                mean_c1[b, :] = np.mean(spike_counts[mask_1, :], axis=0)
            if np.sum(mask_2) > 0:
                mean_c2[b, :] = np.mean(spike_counts[mask_2, :], axis=0)

        # Skip if any position bin has missing data
        if np.isnan(mean_c1).any() or np.isnan(mean_c2).any():
            continue

        region_activity_c1[region].append(mean_c1)
        region_activity_c2[region].append(mean_c2)

        # Fit session-level PCA simultaneously over Context 1 and Context 2
        X_ses_joint = np.vstack([mean_c1, mean_c2])  # (2 * n_bins, n_neurons)
        scaler_ses = StandardScaler()
        scaled_ses = scaler_ses.fit_transform(X_ses_joint)

        n_comp_ses = min(N_COMPONENTS, spike_counts.shape[1], 2 * n_bins)
        pca_ses = PCA(n_components=n_comp_ses)
        proj_ses_joint = pca_ses.fit_transform(scaled_ses)

        # Separate Context 1 and Context 2 projections
        proj_ses_c1 = proj_ses_joint[:n_bins, :]
        proj_ses_c2 = proj_ses_joint[n_bins:, :]

        # Session-level Euclidean distance in shared PC space
        ses_eucl_dist = np.linalg.norm(proj_ses_c1 - proj_ses_c2, axis=1)

        for b in range(n_bins):
            session_pca_records.append({
                'subject': this_subject,
                'session': this_ses,
                'probe': this_probe,
                'region': region,
                'is_far': is_far,
                'position': unique_positions_cm[b],
                'pc1_c1': proj_ses_c1[b, 0],
                'pc2_c1': proj_ses_c1[b, 1],
                'pc1_c2': proj_ses_c2[b, 0],
                'pc2_c2': proj_ses_c2[b, 1],
                'eucl_dist_pc': ses_eucl_dist[b],
                'var_exp1': pca_ses.explained_variance_ratio_[0] * 100,
                'var_exp2': pca_ses.explained_variance_ratio_[1] * 100 if n_comp_ses > 1 else np.nan,
                'total_var_exp': np.sum(pca_ses.explained_variance_ratio_) * 100
            })

session_pca_df = pd.DataFrame(session_pca_records)

# %% Compute Pseudo-Population PCA (Jointly Fitted across Context 1 & Context 2)

pca_pseudo_results = {}

for region in REGION_ORDER:
    if len(region_activity_c1[region]) == 0:
        continue

    # Concatenate all neurons for this brain region across sessions
    X1_region = np.hstack(region_activity_c1[region])  # (n_bins x total_neurons)
    X2_region = np.hstack(region_activity_c2[region])  # (n_bins x total_neurons)

    # Stack Context 1 and Context 2 vertically to fit jointly in the same space
    X_joint = np.vstack([X1_region, X2_region])        # (2 * n_bins x total_neurons)

    # Standardize neurons across all bins and contexts
    scaler = StandardScaler()
    X_joint_scaled = scaler.fit_transform(X_joint)

    # Fit PCA jointly on Context 1 and Context 2
    n_comp = min(N_COMPONENTS, X_joint.shape[1], 2 * n_bins)
    pca = PCA(n_components=n_comp)
    traj_joint = pca.fit_transform(X_joint_scaled)     # (2 * n_bins x n_comp)

    # Separate Context 1 and Context 2 in the shared PC space
    traj_c1 = traj_joint[:n_bins, :]                   # (n_bins x n_comp)
    traj_c2 = traj_joint[n_bins:, :]                   # (n_bins x n_comp)

    # Calculate Euclidean distance between Context 1 and Context 2 in shared 10D PC space
    eucl_dist_10d = np.linalg.norm(traj_c1 - traj_c2, axis=1)  # (n_bins,)

    pca_pseudo_results[region] = {
        'traj_c1': traj_c1,
        'traj_c2': traj_c2,
        'eucl_dist_10d': eucl_dist_10d,
        'var_exp': pca.explained_variance_ratio_ * 100,
        'total_var': np.sum(pca.explained_variance_ratio_) * 100,
        'n_neurons': X_joint.shape[1],
        'n_components': n_comp
    }

# %% Figure 1: 2D State-Space Trajectories (First 2 PCs, Shared Space) per Brain Region

f, axs = plt.subplots(1, len(REGION_ORDER), figsize=(1.6 * len(REGION_ORDER), 1.8), dpi=dpi)
if len(REGION_ORDER) == 1:
    axs = [axs]

for i, region in enumerate(REGION_ORDER):
    if region not in pca_pseudo_results:
        axs[i].axis('off')
        continue

    res = pca_pseudo_results[region]
    traj1 = res['traj_c1']
    traj2 = res['traj_c2']
    v = res['var_exp']
    n_neu = res['n_neurons']

    # Plot Context 1 trajectory (first 2 PCs)
    axs[i].plot(traj1[:, 0], traj1[:, 1], color=colors['context1'], lw=1.5,
                label='Context 1', zorder=2)
    # Start point
    axs[i].scatter(traj1[idx_start, 0], traj1[idx_start, 1], color=colors['context1'],
                   marker='o', s=20, edgecolor='k', lw=0.5, zorder=3)
    # First landmark
    axs[i].scatter(traj1[idx_first_obj, 0], traj1[idx_first_obj, 1], color=colors['context1'],
                   marker='s', s=20, edgecolor='k', lw=0.5, zorder=3)
    # Second landmark
    axs[i].scatter(traj1[idx_far_obj, 0], traj1[idx_far_obj, 1], color=colors['context1'],
                   marker='^', s=25, edgecolor='k', lw=0.5, zorder=3)

    # Plot Context 2 trajectory (first 2 PCs)
    axs[i].plot(traj2[:, 0], traj2[:, 1], color=colors['context2'], lw=1.5,
                label='Context 2', zorder=2)
    # Start point
    axs[i].scatter(traj2[idx_start, 0], traj2[idx_start, 1], color=colors['context2'],
                   marker='o', s=20, edgecolor='k', lw=0.5, zorder=3)
    # First landmark
    axs[i].scatter(traj2[idx_first_obj, 0], traj2[idx_first_obj, 1], color=colors['context2'],
                   marker='s', s=20, edgecolor='k', lw=0.5, zorder=3)
    # Second landmark
    axs[i].scatter(traj2[idx_far_obj, 0], traj2[idx_far_obj, 1], color=colors['context2'],
                   marker='^', s=25, edgecolor='k', lw=0.5, zorder=3)

    axs[i].axhline(0, color='grey', ls='--', lw=0.5, zorder=1)
    axs[i].axvline(0, color='grey', ls='--', lw=0.5, zorder=1)

    region_color = colors.get(region, 'k')
    axs[i].set_title(f'{region}\n(N={n_neu})', color=region_color, weight='bold', fontsize=7)
    axs[i].set_xlabel(f'PC 1 ({v[0]:.1f}%)', labelpad=2)
    if i == 0:
        axs[i].set_ylabel(f'PC 2 ({v[1]:.1f}%)', labelpad=2)

sns.despine(trim=True)
plt.tight_layout()

# Save 2D trajectory figure
save_dir = path_dict['paper_fig_path'] / 'Population'
save_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(save_dir / 'pca_trajectories_2d_per_region.pdf')
plt.savefig(save_dir / 'pca_trajectories_2d_per_region.jpg', dpi=600)
plt.show()

# %% Figure 2: PC1 and PC2 Trajectories vs Track Position (Shared Space)

f, axs = plt.subplots(2, len(REGION_ORDER), figsize=(1.6 * len(REGION_ORDER), 3.2), dpi=dpi, sharex=True)

for i, region in enumerate(REGION_ORDER):
    if region not in pca_pseudo_results:
        axs[0, i].axis('off')
        axs[1, i].axis('off')
        continue

    res = pca_pseudo_results[region]
    traj1 = res['traj_c1']
    traj2 = res['traj_c2']
    v = res['var_exp']

    # --- Row 0: PC1 vs Position ---
    axs[0, i].plot(unique_positions_cm, traj1[:, 0], color=colors['context1'],
                   lw=1.2, label='Context 1')
    axs[0, i].plot(unique_positions_cm, traj2[:, 0], color=colors['context2'],
                   lw=1.2, label='Context 2')
    
    # Landmark spans
    axs[0, i].axvspan(FIRST_OBJ, FIRST_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)
    axs[0, i].axvspan(FAR_OBJ, FAR_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)
    axs[0, i].axhline(0, color='grey', ls='--', lw=0.5)
    
    region_color = colors.get(region, 'k')
    axs[0, i].set_title(region, color=region_color, weight='bold', fontsize=7)
    axs[0, i].legend(frameon=False, fontsize=5.5, loc='upper right')
    if i == 0:
        axs[0, i].set_ylabel(f'PC 1 ({v[0]:.1f}%)')

    # --- Row 1: PC2 vs Position ---
    axs[1, i].plot(unique_positions_cm, traj1[:, 1], color=colors['context1'],
                   lw=1.2, label='Context 1')
    axs[1, i].plot(unique_positions_cm, traj2[:, 1], color=colors['context2'],
                   lw=1.2, label='Context 2')
    
    axs[1, i].axvspan(FIRST_OBJ, FIRST_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)
    axs[1, i].axvspan(FAR_OBJ, FAR_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)
    axs[1, i].axhline(0, color='grey', ls='--', lw=0.5)
    
    axs[1, i].set(xticks=[0, 50, 100, 150], xlim=[0, 150])
    axs[1, i].legend(frameon=False, fontsize=5.5, loc='upper right')
    if i == 0:
        axs[1, i].set_ylabel(f'PC 2 ({v[1]:.1f}%)')

f.supxlabel('Position (cm)', fontsize=7, y=0.03)
sns.despine(trim=True)
plt.tight_layout()

# Save 1D PC trajectory figure
plt.savefig(save_dir / 'pca_pc1_pc2_vs_position_per_region.pdf')
plt.savefig(save_dir / 'pca_pc1_pc2_vs_position_per_region.jpg', dpi=600)
plt.show()

# %% Figure 3: Euclidean Distance in 10D Shared PC Space between Context 1 and 2 vs Position

f, axs = plt.subplots(1, len(REGION_ORDER), figsize=(1.6 * len(REGION_ORDER), 1.8), dpi=dpi, sharey=True)
if len(REGION_ORDER) == 1:
    axs = [axs]

for i, region in enumerate(REGION_ORDER):
    if region not in pca_pseudo_results:
        axs[i].axis('off')
        continue

    res = pca_pseudo_results[region]
    dist_10d = res['eucl_dist_10d']
    n_comp = res['n_components']
    region_color = colors.get(region, 'k')

    # Plot Euclidean distance over position
    axs[i].plot(unique_positions_cm, dist_10d, color=region_color, lw=1.5)

    # Shaded landmark positions
    axs[i].axvspan(FIRST_OBJ, FIRST_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)
    axs[i].axvspan(FAR_OBJ, FAR_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)
    axs[i].axhline(0, color='grey', ls='--', lw=0.5)

    axs[i].set_title(region, color=region_color, weight='bold', fontsize=7)
    axs[i].set(xticks=[0, 50, 100, 150], xlim=[0, 150], xlabel='')
    if i == 0:
        axs[i].set_ylabel(f'Euclidean dist ({n_comp}D PC)', labelpad=2)

f.supxlabel('Position (cm)', fontsize=7, y=0.04)
sns.despine(trim=True)
plt.tight_layout()

# Save Euclidean distance figure
plt.savefig(save_dir / 'pca_eucl_dist_10d_per_region.pdf')
plt.savefig(save_dir / 'pca_eucl_dist_10d_per_region.jpg', dpi=600)
plt.show()
