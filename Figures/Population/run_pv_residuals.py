# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 19/08/2026
"""
# %%

import pickle
import numpy as np
import pandas as pd
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
with open(path_dict['google_drive_data_path'] / 'residuals_position_50mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Load recording info
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv')
rec['date'] = rec['date'].astype(str)

# Loop over recordings
records = []
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

        # Calculate PV correlation for each spatial bin (context 1 vs context 2)
        pv_corr_bin = np.zeros(n_bins)
        for b in range(n_bins):
            v1 = pv_mean_1[b, :]
            v2 = pv_mean_2[b, :]
            if np.std(v1) > 0 and np.std(v2) > 0:
                pv_corr_bin[b] = np.corrcoef(v1, v2)[0, 1]
            else:
                pv_corr_bin[b] = np.nan

        # Calculate PV angle in degrees for each spatial bin
        dot_products = np.sum(pv_mean_1 * pv_mean_2, axis=1)
        norms_1 = np.linalg.norm(pv_mean_1, axis=1)
        norms_2 = np.linalg.norm(pv_mean_2, axis=1)
        norm_product = norms_1 * norms_2
        cosine_sim = np.where(norm_product > 0, dot_products / norm_product, np.nan)
        pv_angle_bin = np.degrees(np.arccos(np.clip(cosine_sim, -1.0, 1.0)))

        # Append per-bin records
        for b, bin_idx in enumerate(rel_pos_bins):
            records.append({
                'subject': int(rec.loc[i, 'subject']),
                'date': rec.loc[i, 'date'],
                'probe': rec.loc[i, 'probe'],
                'region': region,
                'position': bin_idx / 10,  # position in cm
                'pv_corr': pv_corr_bin[b],
                'pv_angle': pv_angle_bin[b],
                'n_neurons': spike_counts.shape[1]
            })

pv_df = pd.DataFrame(records)

# %% Plot PV Correlation - Far Subjects
far_subjects = subjects.loc[subjects['Far'] == 1, 'SubjectID'].values.astype(int)
near_subjects = subjects.loc[subjects['Far'] == 0, 'SubjectID'].values.astype(int)

far_df = pv_df[np.isin(pv_df['subject'], far_subjects)]
regions_far = far_df['region'].unique()

if len(regions_far) > 0:
    f, ax = plt.subplots(1, len(regions_far), figsize=(2.2 * len(regions_far), 2), dpi=dpi, sharey=True, squeeze=False)
    ax = ax.flatten()

    for i, region in enumerate(regions_far):
        region_df = far_df[far_df['region'] == region]
        n_rec = len(region_df.groupby(['subject', 'date']))
        region_color = colors.get(region, 'tab:blue')

        sns.lineplot(data=region_df, x='position', y='pv_corr', color=region_color,
                     errorbar='se', ax=ax[i], err_kws={'lw': 0})
        ax[i].axhline(0, ls='--', color='grey', lw=0.5)

        # Object positions
        ax[i].axvspan(FIRST_OBJ, FIRST_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)
        ax[i].axvspan(FAR_OBJ, FAR_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)

        ax[i].set(title=f'{region} (n={n_rec})', xlabel='Position (cm)', ylabel='PV correlation' if i == 0 else '',
                  xlim=[0, 150], xticks=[0, 50, 100, 150])

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path_dict['fig_path'] / 'pv_correlation_line_far.pdf')
    plt.show()

# %% Plot PV Correlation - Near Subjects
near_df = pv_df[np.isin(pv_df['subject'], near_subjects)]
regions_near = near_df['region'].unique()

if len(regions_near) > 0:
    f, ax = plt.subplots(1, len(regions_near), figsize=(2.2 * len(regions_near), 2), dpi=dpi, sharey=True, squeeze=False)
    ax = ax.flatten()

    for i, region in enumerate(regions_near):
        region_df = near_df[near_df['region'] == region]
        n_rec = len(region_df.groupby(['subject', 'date']))
        region_color = colors.get(region, 'tab:blue')

        sns.lineplot(data=region_df, x='position', y='pv_corr', color=region_color,
                     errorbar='se', ax=ax[i], err_kws={'lw': 0})
        ax[i].axhline(0, ls='--', color='grey', lw=0.5)

        # Object positions
        ax[i].axvspan(FIRST_OBJ, FIRST_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)
        ax[i].axvspan(NEAR_OBJ, NEAR_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)

        ax[i].set(title=f'{region} (n={n_rec})', xlabel='Position (cm)', ylabel='PV correlation' if i == 0 else '',
                  xlim=[0, 150], xticks=[0, 50, 100, 150])

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path_dict['fig_path'] / 'pv_correlation_line_near.pdf')
    plt.show()

# %% Plot PV Angle - Far Subjects
if len(regions_far) > 0:
    f, ax = plt.subplots(1, len(regions_far), figsize=(2.2 * len(regions_far), 2), dpi=dpi, sharey=True, squeeze=False)
    ax = ax.flatten()

    for i, region in enumerate(regions_far):
        region_df = far_df[far_df['region'] == region]
        n_rec = len(region_df.groupby(['subject', 'date']))
        region_color = colors.get(region, 'tab:blue')

        sns.lineplot(data=region_df, x='position', y='pv_angle', color=region_color,
                     errorbar='se', ax=ax[i], err_kws={'lw': 0})

        # Object positions
        ax[i].axvspan(FIRST_OBJ, FIRST_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)
        ax[i].axvspan(FAR_OBJ, FAR_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)

        ax[i].set(title=f'{region} (n={n_rec})', xlabel='Position (cm)', ylabel='PV angle (deg)' if i == 0 else '',
                  xlim=[0, 150], xticks=[0, 50, 100, 150])

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path_dict['fig_path'] / 'pv_angle_line_far.pdf')
    plt.show()

# %% Plot PV Angle - Near Subjects
if len(regions_near) > 0:
    f, ax = plt.subplots(1, len(regions_near), figsize=(2.2 * len(regions_near), 2), dpi=dpi, sharey=True, squeeze=False)
    ax = ax.flatten()

    for i, region in enumerate(regions_near):
        region_df = near_df[near_df['region'] == region]
        n_rec = len(region_df.groupby(['subject', 'date']))
        region_color = colors.get(region, 'tab:blue')

        sns.lineplot(data=region_df, x='position', y='pv_angle', color=region_color,
                     errorbar='se', ax=ax[i], err_kws={'lw': 0})

        # Object positions
        ax[i].axvspan(FIRST_OBJ, FIRST_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)
        ax[i].axvspan(NEAR_OBJ, NEAR_OBJ + OBJ_SIZE, color='gray', alpha=0.25, lw=0)

        ax[i].set(title=f'{region} (n={n_rec})', xlabel='Position (cm)', ylabel='PV angle (deg)' if i == 0 else '',
                  xlim=[0, 150], xticks=[0, 50, 100, 150])

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path_dict['fig_path'] / 'pv_angle_line_near.pdf')
    plt.show()
