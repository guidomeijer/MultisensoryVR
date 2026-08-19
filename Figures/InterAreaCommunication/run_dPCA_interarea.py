# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:28:35 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy import stats
from itertools import combinations
import mne
import matplotlib.pyplot as plt
from dPCA import dPCA
from msvr_functions import paths, load_subjects, figure_style, load_objects, add_significance

colors, dpi = figure_style()
mne.set_log_level('WARNING')

# Settings
MIN_NEURONS = 5
USE_TYPE = 'ALL'  # INT, PYR or ALL
BASELINE_WIN = [200, 400]


def run_stats_vs_zero(df):
    """Cluster-based permutation test against zero across sessions."""
    df = df.copy()
    df['position'] = pd.to_numeric(df['position'])
    test_matrix = df.pivot_table(
        index=['subject', 'session'], columns='position', values='correlation', aggfunc='mean')
    
    if test_matrix.shape[0] < 3:
        return np.ones(test_matrix.shape[1]), test_matrix.columns.values
        
    positions = test_matrix.columns.values
    X = test_matrix.values
    
    # Calculate threshold
    t_threshold = stats.t.ppf(1 - 0.05 / 2, test_matrix.shape[0] - 1)
    
    # Run MNE cluster test
    t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        X,
        threshold=t_threshold,
        n_permutations=1000, 
        tail=0,          
        out_type='mask'  
    )
    
    p_values = np.ones(len(positions))
    for cluster_mask, p_val in zip(clusters, cluster_p_values):
        p_values[cluster_mask] = p_val
        
    return p_values, positions


# Load in data
path_dict = paths()
subjects = load_subjects()    
with open(path_dict['google_drive_data_path'] / 'residuals_position_20mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Add neuron type to spike_dict
neuron_type = pd.read_csv(path_dict['save_path'] / 'waveform_metrics.csv',
                          dtype={'subject': str, 'date': str})

# Group recordings by date (simultaneous recordings)
unique_dates = np.unique(spike_dict['date'])

corr_results = []
for date in unique_dates:

    # Find all recording indices for this date
    rec_indices = [idx for idx in range(len(spike_dict['date'])) if spike_dict['date'][idx] == date]
    this_subject = spike_dict['subject'][rec_indices[0]]
    
    # Get whether this is a FAR or NEAR session
    is_far = subjects.loc[subjects['SubjectID'] == this_subject, 'Far'].values[0]
    
    # Load in object data
    obj_df = load_objects(this_subject, str(date))

    # Get which context is the rewarded context for the first and second object
    obj1_goal = obj_df.loc[(obj_df['object'] == 1) & (obj_df['goal'] == 1), 'sound'].values[0]
    obj2_goal = obj_df.loc[(obj_df['object'] == 2) & (obj_df['goal'] == 1), 'sound'].values[0]

    # Find all unique regions recorded on this date across all probes
    session_regions = []
    for idx in rec_indices:
        session_regions.extend(np.unique(spike_dict['region'][idx]))
    session_regions = [r for r in np.unique(session_regions) if r != 'root']

    # Project neural activity onto context projection for each brain region
    region_latents = {}
    spatial_bins = spike_dict['position'][rec_indices[0]]
    context_per_bin = spike_dict['context'][rec_indices[0]]
    unique_positions = np.unique(spatial_bins)
    n_bins = len(unique_positions)

    n_trials_A = spatial_bins[context_per_bin == obj1_goal].shape[0] // n_bins
    n_trials_B = spatial_bins[context_per_bin == 3 - obj1_goal].shape[0] // n_bins
    min_trials = np.min([n_trials_A, n_trials_B])
    if min_trials < 2:
        continue

    for region in session_regions:
        # Collect neurons for this region across probes
        region_spikes = []
        for idx in rec_indices:
            this_probe = spike_dict['probe'][idx]
            if USE_TYPE != 'ALL':
                neuron_types = neuron_type[(neuron_type['subject'] == this_subject) &
                                           (neuron_type['date'] == str(date)) &
                                           (neuron_type['probe'] == this_probe) &
                                           (neuron_type['unit_id'].isin(spike_dict['neuron_id'][idx]))]['neuron_type'].values
            else:
                neuron_types = np.array(['ALL'] * len(spike_dict['neuron_id'][idx]))

            mask = (spike_dict['region'][idx] == region) & (neuron_types == USE_TYPE)
            if np.sum(mask) > 0:
                region_spikes.append(spike_dict['residuals'][idx][:, mask])

        if len(region_spikes) == 0:
            continue
        spike_counts = np.hstack(region_spikes)

        # Throw out silent neurons
        spike_counts = spike_counts[:, np.std(spike_counts, axis=0) > 0.5]

        # If not enough neurons, continue
        if spike_counts.shape[1] < MIN_NEURONS:
            continue

        # Z-score the spike counts
        spike_counts = (spike_counts - np.mean(spike_counts, axis=0)) / np.std(spike_counts, axis=0)
        n_neurons = spike_counts.shape[1]

        # Create arrays per context (neurons x spatial bins x trials)
        state_A_trials = spike_counts[context_per_bin == obj1_goal, :].reshape(
            n_trials_A, n_bins, n_neurons).transpose(2, 1, 0)[:, :, :min_trials]
        state_B_trials = spike_counts[context_per_bin == 3 - obj1_goal, :].reshape(
            n_trials_B, n_bins, n_neurons).transpose(2, 1, 0)[:, :, :min_trials]

        # Create 4D array (neurons x state x spatial bins x trials)
        X_trials = np.zeros((n_neurons, 2, n_bins, min_trials))
        X_trials[:, 0, :, :] = state_A_trials
        X_trials[:, 1, :, :] = state_B_trials

        # Create 3D trial-averaged array
        X_mean = np.nanmean(X_trials, axis=-1)

        # Initialize and fit dPCA
        dpca = dPCA.dPCA(labels='st', regularizer=0.01)
        dpca.protect = ['t']
        dpca.fit(X_mean, X_trials)

        # Project single trials onto context axis ('s')
        Z_trials = dpca.transform(X_trials)
        Z_mean = dpca.transform(X_mean)

        s_trials = Z_trials['s'][0]  # shape: (2 states, n_bins, min_trials)

        # Align sign so Context 1 is consistently positive relative to Context 2
        if np.nanmean(Z_mean['s'][0, 0, :]) < np.nanmean(Z_mean['s'][0, 1, :]):
            s_trials = -s_trials

        # Store single-trial context latents per spatial bin: shape (n_bins, 2 * min_trials)
        region_latents[region] = np.concatenate((s_trials[0, :, :], s_trials[1, :, :]), axis=1)

    # Correlate latent context projections per spatial bin for each pair of simultaneously recorded regions
    available_regions = list(region_latents.keys())
    if len(available_regions) < 2:
        continue

    for region1, region2 in combinations(available_regions, 2):
        if region1 > region2:
            region1, region2 = region2, region1
        pair_name = f'{region1}-{region2}'

        r_per_bin = np.zeros(len(unique_positions))
        for b, pos in enumerate(unique_positions):
            z1 = region_latents[region1][b, :]
            z2 = region_latents[region2][b, :]

            if np.std(z1) > 0 and np.std(z2) > 0:
                r_per_bin[b] = stats.pearsonr(z1, z2)[0]
            else:
                r_per_bin[b] = np.nan

        # Baseline subtraction (0 to 400 mm)
        baseline_mask = (unique_positions >= BASELINE_WIN[0]) & (unique_positions <= BASELINE_WIN[1])
        baseline_r = np.nanmean(r_per_bin[baseline_mask])
        r_per_bin = r_per_bin - baseline_r

        for b, pos in enumerate(unique_positions):
            corr_results.append({
                'subject': this_subject,
                'session': date,
                'region1': region1,
                'region2': region2,
                'region_pair': pair_name,
                'position': pos,
                'correlation': r_per_bin[b],
                'is_far': is_far
            })

corr_df = pd.DataFrame(corr_results)

# %% Plot Trial-by-Trial Context Correlation for FAR Sessions
far_df = corr_df[corr_df['is_far'] == 1]
unique_pairs_far = sorted(far_df['region_pair'].unique())
n_pairs_far = len(unique_pairs_far)

if n_pairs_far > 0:
    n_cols = 5
    n_rows = int(np.ceil(n_pairs_far / n_cols))
    f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5), dpi=dpi, sharey=True, sharex=True)
    axs = np.array(axs).flatten()

    for idx, pair in enumerate(unique_pairs_far):
        ax = axs[idx]
        plot_df = far_df[far_df['region_pair'] == pair]

        sns.lineplot(data=plot_df, x='position', y='correlation', ax=ax,
                     color='k', errorbar='se', err_kws={'lw': 0}, zorder=1)

        if len(plot_df['session'].unique()) >= 3:
            p_values, positions = run_stats_vs_zero(plot_df)
            add_significance(positions, p_values, ax=ax)

        ax.axhline(0, ls='--', lw=0.5, color='gray', zorder=0)
        ax.plot([425, 425], [-0.5, 0.5], ls='--', lw=0.5, color='k', zorder=0)
        ax.plot([1325, 1325], [-0.5, 0.5], ls='--', lw=0.5, color='k', zorder=0)
        ax.set(title=pair, xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
               xlabel='', ylabel='', ylim=[-0.25, 0.25])

    # Hide unused axes
    for idx in range(n_pairs_far, len(axs)):
        axs[idx].axis('off')

    axs[0].set_ylabel('Context latent correlation', labelpad=0)
    f.supxlabel('Position (cm)', fontsize=7, y=0.04)
    f.suptitle('FAR', fontsize=8)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path_dict['paper_fig_path'] / 'dPCA' / f'dpca_context_corr_far_{USE_TYPE}.pdf')
    plt.savefig(path_dict['paper_fig_path'] / 'dPCA' / f'dpca_context_corr_far_{USE_TYPE}.jpg', dpi=600)
    plt.show()

# %% Plot Trial-by-Trial Context Correlation for NEAR Sessions
near_df = corr_df[corr_df['is_far'] == 0]
unique_pairs_near = sorted(near_df['region_pair'].unique())
n_pairs_near = len(unique_pairs_near)

if n_pairs_near > 0:
    n_cols = 5
    n_rows = int(np.ceil(n_pairs_near / n_cols))
    f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5), dpi=dpi, sharey=True, sharex=True)
    axs = np.array(axs).flatten()

    for idx, pair in enumerate(unique_pairs_near):
        ax = axs[idx]
        plot_df = near_df[near_df['region_pair'] == pair]

        sns.lineplot(data=plot_df, x='position', y='correlation', ax=ax,
                     color='k', errorbar='se', err_kws={'lw': 0}, zorder=1)

        if len(plot_df['session'].unique()) >= 3:
            p_values, positions = run_stats_vs_zero(plot_df)
            add_significance(positions, p_values, ax=ax)

        ax.axhline(0, ls='--', lw=0.5, color='gray', zorder=0)
        ax.plot([425, 425], [-0.5, 0.5], ls='--', lw=0.5, color='k', zorder=0)
        ax.plot([875, 875], [-0.5, 0.5], ls='--', lw=0.5, color='k', zorder=0)
        ax.set(title=pair, xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
               xlabel='', ylabel='')

    # Hide unused axes
    for idx in range(n_pairs_near, len(axs)):
        axs[idx].axis('off')

    axs[0].set_ylabel('Context latent correlation', labelpad=0)
    f.supxlabel('Position (cm)', fontsize=7, y=0.04)
    f.suptitle('NEAR', fontsize=8)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path_dict['paper_fig_path'] / 'dPCA' / f'dpca_context_corr_near_{USE_TYPE}.pdf')
    plt.savefig(path_dict['paper_fig_path'] / 'dPCA' / f'dpca_context_corr_near_{USE_TYPE}.jpg', dpi=600)
    plt.show()


