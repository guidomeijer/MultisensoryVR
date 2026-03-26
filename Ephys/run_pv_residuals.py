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
MIN_NEURONS = 10
FIRST_OBJ = 45
FAR_OBJ = 135
NEAR_OBJ = 90
OBJ_SIZE = 5

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

        # Add to dataframe
        pv_corr_df = pd.concat([pv_corr_df, pd.DataFrame([{
            'subject': rec.loc[i, 'subject'],
            'date': rec.loc[i, 'date'],
            'probe': rec.loc[i, 'probe'],
            'region': region,
            'pv_corr': pv_corr,
            'pv_angle': pv_angle,
            'n_neurons': spike_counts.shape[1]
        }])], ignore_index=True)

# %% Plot
rel_pos_bins_m = rel_pos_bins / 10
regions = pv_corr_df['region'].unique()
f, ax = plt.subplots(1, len(regions), figsize=(2 * len(regions), 2), dpi=dpi, squeeze=False)
ax = ax.flatten()

for i, region in enumerate(regions):
    region_df = pv_corr_df[(pv_corr_df['region'] == region)
                            & np.isin(pv_corr_df['subject'].values,
                                      subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
    pv_corr = np.mean(np.stack(region_df['pv_corr'].values), axis=0)
    sns.heatmap(pv_corr, cmap='coolwarm', vmin=-0.8, vmax=0.8, square=True, ax=ax[i], cbar=False)
    ax[i].plot([np.abs(rel_pos_bins_m - FIRST_OBJ).argmin(), np.abs(rel_pos_bins_m - FIRST_OBJ).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - FIRST_OBJ).argmin(), np.abs(rel_pos_bins_m - FIRST_OBJ).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - FAR_OBJ).argmin(), np.abs(rel_pos_bins_m - FAR_OBJ).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - FAR_OBJ).argmin(), np.abs(rel_pos_bins_m - FAR_OBJ).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - (FAR_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FAR_OBJ + OBJ_SIZE)).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - (FAR_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FAR_OBJ + OBJ_SIZE)).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].set(title=f'{region} (n={region_df.shape[0]})', xticks=[], yticks=[])

plt.tight_layout()
plt.savefig(path_dict['fig_path'] / 'pv_correlation_far.pdf')
plt.show()

# %%
f, ax = plt.subplots(1, len(regions), figsize=(2 * len(regions), 2), dpi=dpi, squeeze=False)
ax = ax.flatten()

for i, region in enumerate(regions):
    region_df = pv_corr_df[(pv_corr_df['region'] == region)
                            & np.isin(pv_corr_df['subject'].values,
                                      subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int))]
    pv_corr = np.mean(np.stack(region_df['pv_corr'].values), axis=0)
    sns.heatmap(pv_corr, cmap='coolwarm', vmin=-0.8, vmax=0.8, square=True, ax=ax[i], cbar=False)
    ax[i].plot([np.abs(rel_pos_bins_m - FIRST_OBJ).argmin(), np.abs(rel_pos_bins_m - FIRST_OBJ).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - FIRST_OBJ).argmin(), np.abs(rel_pos_bins_m - FIRST_OBJ).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - NEAR_OBJ).argmin(), np.abs(rel_pos_bins_m - NEAR_OBJ).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - NEAR_OBJ).argmin(), np.abs(rel_pos_bins_m - NEAR_OBJ).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - (NEAR_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (NEAR_OBJ + OBJ_SIZE)).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - (NEAR_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (NEAR_OBJ + OBJ_SIZE)).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].set(title=f'{region} (n={region_df.shape[0]})', xticks=[], yticks=[])

plt.tight_layout()
plt.savefig(path_dict['fig_path'] / 'pv_correlation_near.pdf')
plt.show()

# %% Plot
rel_pos_bins_m = rel_pos_bins / 10
regions = pv_corr_df['region'].unique()
f, ax = plt.subplots(1, len(regions), figsize=(2 * len(regions), 2), dpi=dpi, squeeze=False)
ax = ax.flatten()

for i, region in enumerate(regions):
    region_df = pv_corr_df[(pv_corr_df['region'] == region)
                            & np.isin(pv_corr_df['subject'].values,
                                      subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))]
    pv_corr = np.mean(np.stack(region_df['pv_angle'].values), axis=0)
    sns.heatmap(pv_corr, cmap='magma', vmin=0, vmax=90, square=True, ax=ax[i], cbar=False)
    ax[i].plot([np.abs(rel_pos_bins_m - FIRST_OBJ).argmin(), np.abs(rel_pos_bins_m - FIRST_OBJ).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - FIRST_OBJ).argmin(), np.abs(rel_pos_bins_m - FIRST_OBJ).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - FAR_OBJ).argmin(), np.abs(rel_pos_bins_m - FAR_OBJ).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - FAR_OBJ).argmin(), np.abs(rel_pos_bins_m - FAR_OBJ).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - (FAR_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FAR_OBJ + OBJ_SIZE)).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - (FAR_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FAR_OBJ + OBJ_SIZE)).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].set(title=f'{region} (n={region_df.shape[0]})', xticks=[], yticks=[])

plt.tight_layout()
plt.savefig(path_dict['fig_path'] / 'pv_angle_far.pdf')
plt.show()

# %%
f, ax = plt.subplots(1, len(regions), figsize=(2 * len(regions), 2), dpi=dpi, squeeze=False)
ax = ax.flatten()

for i, region in enumerate(regions):
    region_df = pv_corr_df[(pv_corr_df['region'] == region)
                            & np.isin(pv_corr_df['subject'].values,
                                      subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int))]
    pv_corr = np.mean(np.stack(region_df['pv_angle'].values), axis=0)
    sns.heatmap(pv_corr, cmap='magma', vmin=0, vmax=90, square=True, ax=ax[i], cbar=False)
    ax[i].plot([np.abs(rel_pos_bins_m - FIRST_OBJ).argmin(), np.abs(rel_pos_bins_m - FIRST_OBJ).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - FIRST_OBJ).argmin(), np.abs(rel_pos_bins_m - FIRST_OBJ).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - NEAR_OBJ).argmin(), np.abs(rel_pos_bins_m - NEAR_OBJ).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - NEAR_OBJ).argmin(), np.abs(rel_pos_bins_m - NEAR_OBJ).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].plot([np.abs(rel_pos_bins_m - (NEAR_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (NEAR_OBJ + OBJ_SIZE)).argmin()],
               [0, pv_corr.shape[0]], color='white', linestyle='--', lw=0.5)
    ax[i].plot([0, pv_corr.shape[0]], [np.abs(rel_pos_bins_m - (NEAR_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (NEAR_OBJ + OBJ_SIZE)).argmin()],
               color='white', linestyle='--', lw=0.5)
    ax[i].set(title=f'{region} (n={region_df.shape[0]})', xticks=[], yticks=[])

plt.tight_layout()
plt.savefig(path_dict['fig_path'] / 'pv_angle_near.pdf')
plt.show()

# %% Plot all individual PV correlation matrices

for i in pv_corr_df.index:

    subject = pv_corr_df.loc[i, 'subject']
    date = pv_corr_df.loc[i, 'date']
    region = pv_corr_df.loc[i, 'region']
    far = np.isin(subject, subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int))
    if far:
        second_obj = FAR_OBJ
    else:
        second_obj = NEAR_OBJ

    f, ax = plt.subplots(figsize=(2, 2), dpi=dpi)
    sns.heatmap(pv_corr_df.loc[i, 'pv_corr'], cmap='coolwarm', vmin=-0.8, vmax=0.8, square=True, ax=ax, cbar=False)
    ax.plot([np.abs(rel_pos_bins_m - FIRST_OBJ).argmin(), np.abs(rel_pos_bins_m - FIRST_OBJ).argmin()],
            [0, rel_pos_bins_m.shape[0]], color='white', linestyle='--', lw=0.5)
    ax.plot([0, rel_pos_bins_m.shape[0]], [np.abs(rel_pos_bins_m - FIRST_OBJ).argmin(), np.abs(rel_pos_bins_m - FIRST_OBJ).argmin()],
            color='white', linestyle='--', lw=0.5)
    ax.plot([np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin()],
            [0, rel_pos_bins_m.shape[0]], color='white', linestyle='--', lw=0.5)
    ax.plot([0, rel_pos_bins_m.shape[0]], [np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (FIRST_OBJ + OBJ_SIZE)).argmin()],
            color='white', linestyle='--', lw=0.5)
    ax.plot([np.abs(rel_pos_bins_m - second_obj).argmin(), np.abs(rel_pos_bins_m - second_obj).argmin()],
            [0, rel_pos_bins_m.shape[0]], color='white', linestyle='--', lw=0.5)
    ax.plot([0, rel_pos_bins_m.shape[0]], [np.abs(rel_pos_bins_m - second_obj).argmin(), np.abs(rel_pos_bins_m - second_obj).argmin()],
            color='white', linestyle='--', lw=0.5)
    ax.plot([np.abs(rel_pos_bins_m - (second_obj + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (second_obj + OBJ_SIZE)).argmin()],
            [0, rel_pos_bins_m.shape[0]], color='white', linestyle='--', lw=0.5)
    ax.plot([0, rel_pos_bins_m.shape[0]], [np.abs(rel_pos_bins_m - (second_obj + OBJ_SIZE)).argmin(), np.abs(rel_pos_bins_m - (second_obj + OBJ_SIZE)).argmin()],
              color='white', linestyle='--', lw=0.5)
    ax.set(title=f'{region} (n={pv_corr_df.loc[i, "n_neurons"]} neurons)', xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(path_dict['fig_path'] /  'PV_correlation' / f'{subject}_{date}_{region}.jpg', dpi=600)
    plt.close(f)