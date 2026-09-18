# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 2026

By Guido Meijer

Pairwise noise correlation between brain regions across trials for each position bin
using residual firing rates (residuals_position_20mms.pickle).
"""

import pickle
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from msvr_functions import paths, load_subjects, figure_style

colors, dpi = figure_style()

# Settings
MIN_NEURONS = 1          # Minimum neurons required per region in a session
MIN_TRIALS = 5           # Minimum trials required per context
USE_FAR_ONLY = False     # Set to True if only FAR sessions should be included

# Load paths and subjects
path_dict = paths()
subjects = load_subjects()

# Load in residuals data
with open(path_dict['google_drive_data_path'] / 'residuals_position_20mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Group recordings by date (simultaneously recorded probes)
unique_dates = np.unique(spike_dict['date'])

results = []

for date in unique_dates:
    # Find all recording indices (probes) for this date
    rec_indices = [idx for idx in range(len(spike_dict['date'])) if spike_dict['date'][idx] == date]
    this_subject = spike_dict['subject'][rec_indices[0]]

    # Subject FAR / NEAR status
    is_far = subjects.loc[subjects['SubjectID'] == this_subject, 'Far'].values[0]
    if USE_FAR_ONLY and is_far != 1:
        continue

    # Spatial bins and context per sample
    spatial_bins = spike_dict['position'][rec_indices[0]]
    context_per_bin = spike_dict['context'][rec_indices[0]]
    unique_positions = np.unique(spatial_bins)
    n_bins = len(unique_positions)

    # Determine number of trials per context
    n_trials_ctx1 = np.sum(context_per_bin == 1) // n_bins
    n_trials_ctx2 = np.sum(context_per_bin == 2) // n_bins
    min_trials = min(n_trials_ctx1, n_trials_ctx2)

    if min_trials < MIN_TRIALS:
        continue

    # Find all unique regions recorded on this date across probes
    session_regions = []
    for idx in rec_indices:
        session_regions.extend(np.unique(spike_dict['region'][idx]))
    session_regions = [r for r in np.unique(session_regions) if r != 'root']

    # Extract cleaned neural residuals: (total_trials, n_bins, n_neurons) per region
    region_activity = {}
    for region in session_regions:
        region_spikes = []
        for idx in rec_indices:
            mask = (spike_dict['region'][idx] == region)
            if np.sum(mask) > 0:
                region_spikes.append(spike_dict['residuals'][idx][:, mask])

        if len(region_spikes) == 0:
            continue

        # Stack across probes if multiple probes recorded from this region
        spike_residuals = np.hstack(region_spikes)

        # Throw out silent / invariant neurons
        spike_residuals = spike_residuals[:, np.std(spike_residuals, axis=0) > 0.01]
        if spike_residuals.shape[1] < MIN_NEURONS:
            continue

        n_neurons = spike_residuals.shape[1]

        # Reshape to (trials, n_bins, n_neurons) for each context
        ctx1_trials = spike_residuals[context_per_bin == 1, :].reshape(
            n_trials_ctx1, n_bins, n_neurons)[:min_trials, :, :]
        ctx2_trials = spike_residuals[context_per_bin == 2, :].reshape(
            n_trials_ctx2, n_bins, n_neurons)[:min_trials, :, :]

        # Concatenate trials across contexts: shape (2 * min_trials, n_bins, n_neurons)
        all_trials_activity = np.concatenate([ctx1_trials, ctx2_trials], axis=0)
        region_activity[region] = all_trials_activity

    available_regions = list(region_activity.keys())
    if len(available_regions) < 2:
        continue

    # Compute pairwise correlations between all region pairs (Region A, Region B)
    for region_a, region_b in combinations(available_regions, 2):
        if region_a > region_b:
            region_a, region_b = region_b, region_a
        pair_name = f'{region_a}-{region_b}'

        # Activity arrays: (total_trials, n_bins, n_neurons)
        act_a = region_activity[region_a]
        act_b = region_activity[region_b]
        n_neurons_a = act_a.shape[2]
        n_neurons_b = act_b.shape[2]

        for b_idx, pos in enumerate(unique_positions):
            # Neuronal activity at this position bin across trials:
            # X shape: (trials, n_neurons_a), Y shape: (trials, n_neurons_b)
            X = act_a[:, b_idx, :]
            Y = act_b[:, b_idx, :]

            # Compute correlation over trials for each neuron pair between region A and region B
            # Standardize across trials for vectorized correlation calculation
            X_std = np.std(X, axis=0, keepdims=True)
            Y_std = np.std(Y, axis=0, keepdims=True)

            valid_a = (X_std[0] > 0)
            valid_b = (Y_std[0] > 0)

            if not np.any(valid_a) or not np.any(valid_b):
                mean_corr = np.nan
            else:
                X_valid = X[:, valid_a]
                Y_valid = Y[:, valid_b]

                X_centered = (X_valid - np.mean(X_valid, axis=0, keepdims=True)) / np.std(X_valid, axis=0, keepdims=True)
                Y_centered = (Y_valid - np.mean(Y_valid, axis=0, keepdims=True)) / np.std(Y_valid, axis=0, keepdims=True)

                # Pairwise correlation matrix: (n_valid_neurons_a, n_valid_neurons_b)
                # corr_matrix[i, j] = Pearson r between neuron i in region A and neuron j in region B
                n_trials = X.shape[0]
                corr_matrix = (X_centered.T @ Y_centered) / n_trials

                # Mean over all pairwise correlations between the two brain regions for this spatial bin
                mean_corr = np.nanmean(corr_matrix)

            results.append({
                'subject': this_subject,
                'session': str(date),
                'date': str(date),
                'region_a': region_a,
                'region_b': region_b,
                'region_pair': pair_name,
                'position': pos,
                'position_bin': b_idx,
                'mean_pairwise_corr': mean_corr,
                'n_neurons_a': n_neurons_a,
                'n_neurons_b': n_neurons_b,
                'is_far': is_far
            })

# Convert to DataFrame
corr_df = pd.DataFrame(results)

# 1. Average r values over neuron pairs per spatial bin for each session
session_corr = corr_df.groupby(
    ['subject', 'session', 'region_pair', 'position', 'is_far'],
    as_index=False
)['mean_pairwise_corr'].mean()

# 2. Plot correlation over space for each region pair averaged over sessions
save_fig_dir = path_dict['paper_fig_path'] / 'InterAreaCorrelation'
save_fig_dir.mkdir(parents=True, exist_ok=True)

unique_pairs = sorted(session_corr['region_pair'].unique())
n_pairs = len(unique_pairs)

if n_pairs > 0:
    n_cols = 5
    n_rows = int(np.ceil(n_pairs / n_cols))
    f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.6), dpi=dpi, sharey=True, sharex=True)
    axs = np.atleast_1d(axs).flatten()

    for idx, pair in enumerate(unique_pairs):
        ax = axs[idx]
        pair_df = session_corr[session_corr['region_pair'] == pair]

        # Plot average over sessions with standard error shading
        sns.lineplot(
            data=pair_df,
            x='position',
            y='mean_pairwise_corr',
            ax=ax,
            color='k',
            errorbar='se',
            err_kws={'lw': 0},
            zorder=1
        )

        ax.axhline(0, ls='--', lw=0.5, color='gray', zorder=0)
        # Mark object positions (Object 1 at 425 mm, Far Object 2 at 1325 mm)
        ax.axvline(425, ls='--', lw=0.5, color='k', zorder=0)
        ax.axvline(1325, ls='--', lw=0.5, color='k', zorder=0)

        ax.set(
            title=pair,
            xticks=[0, 500, 1000, 1500],
            xticklabels=[0, 50, 100, 150],
            xlabel='',
            ylabel=''
        )

    # Hide any unused subplots
    for idx in range(n_pairs, len(axs)):
        axs[idx].axis('off')

    axs[0].set_ylabel('Mean pairwise r', labelpad=0)
    f.supxlabel('Position (cm)', fontsize=7, y=0.03)
    f.suptitle('Pairwise Residual Correlation across Space', fontsize=8)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(save_fig_dir / 'pairwise_residuals_correlation.pdf')
    plt.savefig(save_fig_dir / 'pairwise_residuals_correlation.jpg', dpi=600)
    plt.show()

print("Plotting completed.")
