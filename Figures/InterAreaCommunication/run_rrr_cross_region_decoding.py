# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 2026

By Guido Meijer

Cross-Region Predictive Decoding via Reduced Rank Regression (RRR)
Predict the population activity of a target region (e.g., CA1) from a source region
(e.g., TEa or PERI) using low-rank linear regression.

Method:
- Split trials using cross-validation (e.g., Stratified K-Fold across contexts).
- Fit reduced rank regression \hat{Y}_target = X_source * B on training trials.
- Predict target activity on test trials: \hat{Y}_test = X_source_test * B.
- Train a context decoder (LDA) on actual target activity Y_train.
- Evaluate the decoder on the predicted target activity \hat{Y}_test (and actual target activity Y_test).
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy import stats
from itertools import permutations
import mne
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from msvr_functions import paths, load_subjects, figure_style, load_objects, add_significance

colors, dpi = figure_style()
mne.set_log_level('WARNING')

# Settings
MIN_NEURONS = 5
MIN_TRIALS = 10
RRR_RANK = 3           # Dimensionality of the predictive communication subspace
N_SPLITS = 5           # K-fold cross-validation
USE_TYPE = 'ALL'       # INT, PYR or ALL
RIDGE_ALPHA = 1e-4     # Ridge regularization parameter for RRR inversion
RANDOM_STATE = 42


def fit_reduced_rank_regression(X_train, Y_train, rank=3, ridge_alpha=1e-4):
    """
    Fit Reduced Rank Regression (RRR):
        Y = X * B + E, where rank(B) <= rank.
    
    Using the standard analytical SVD solution:
    1. Fit ridge / OLS regression: B_ols = (X^T X + alpha * I)^(-1) X^T Y
    2. Predict \hat{Y}_ols = X * B_ols
    3. Low-rank approximation via SVD on \hat{Y}_ols:
       \hat{Y}_ols = U * S * V^T
       V_r = V[:, :rank]
       B_rrr = B_ols * V_r * V_r^T
    
    Parameters
    ----------
    X_train : np.ndarray (n_samples, n_source_neurons)
        Source area activity (centered).
    Y_train : np.ndarray (n_samples, n_target_neurons)
        Target area activity (centered).
    rank : int
        Subspace rank constraint.
    ridge_alpha : float
        L2 regularization parameter for numerical stability.

    Returns
    -------
    B_rrr : np.ndarray (n_source_neurons, n_target_neurons)
        Low-rank regression weight matrix.
    """
    n_samples, n_src = X_train.shape
    _, n_tgt = Y_train.shape
    eff_rank = min(rank, n_src, n_tgt)

    # Ridge regularized least squares
    XtX = X_train.T @ X_train
    reg = ridge_alpha * np.trace(XtX) / max(n_src, 1) * np.eye(n_src)
    B_ols = np.linalg.solve(XtX + reg, X_train.T @ Y_train)

    # Low rank factor via SVD of predicted activity
    Y_pred_ols = X_train @ B_ols
    try:
        _, _, Vt = np.linalg.svd(Y_pred_ols, full_matrices=False)
        V_r = Vt[:eff_rank, :].T  # (n_tgt, eff_rank)
        B_rrr = B_ols @ V_r @ V_r.T
    except np.linalg.LinAlgError:
        B_rrr = B_ols

    return B_rrr


def run_stats_vs_chance(df, chance_level=0.5):
    """Cluster-based permutation test against chance across sessions."""
    df = df.copy()
    df['position'] = pd.to_numeric(df['position'])
    test_matrix = df.pivot_table(
        index=['subject', 'session'], columns='position', values='accuracy', aggfunc='mean')
    
    if test_matrix.shape[0] < 3:
        return np.ones(test_matrix.shape[1]), test_matrix.columns.values
        
    positions = test_matrix.columns.values
    X = test_matrix.values - chance_level
    
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


# %% Load in data
path_dict = paths()
subjects = load_subjects()    
with open(path_dict['google_drive_data_path'] / 'residuals_position_20mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Add neuron type to spike_dict
neuron_type = pd.read_csv(path_dict['save_path'] / 'waveform_metrics.csv',
                          dtype={'subject': str, 'date': str})

# Group recordings by date (simultaneous recordings)
unique_dates = np.unique(spike_dict['date'])

results = []

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

    spatial_bins = spike_dict['position'][rec_indices[0]]
    context_per_bin = spike_dict['context'][rec_indices[0]]
    unique_positions = np.unique(spatial_bins)
    n_bins = len(unique_positions)

    # Determine trials
    n_trials_A = spatial_bins[context_per_bin == obj1_goal].shape[0] // n_bins
    n_trials_B = spatial_bins[context_per_bin == 3 - obj1_goal].shape[0] // n_bins
    min_trials = np.min([n_trials_A, n_trials_B])
    total_trials = 2 * min_trials
    if min_trials < MIN_TRIALS:
        continue

    # Extract cleaned neural activity (trials x bins x neurons) per region
    region_activity = {}
    for region in session_regions:
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
        if spike_counts.shape[1] < MIN_NEURONS:
            continue

        # Z-score the spike counts across positions & trials
        spike_counts = (spike_counts - np.mean(spike_counts, axis=0)) / np.std(spike_counts, axis=0)
        n_neurons = spike_counts.shape[1]

        # Reshape to (trials, n_bins, n_neurons)
        state_A_trials = spike_counts[context_per_bin == obj1_goal, :].reshape(
            n_trials_A, n_bins, n_neurons)[:min_trials, :, :]
        state_B_trials = spike_counts[context_per_bin == 3 - obj1_goal, :].reshape(
            n_trials_B, n_bins, n_neurons)[:min_trials, :, :]

        # Concatenate: first min_trials are context 1, next min_trials are context 2
        all_trials_activity = np.concatenate([state_A_trials, state_B_trials], axis=0)
        region_activity[region] = all_trials_activity

    available_regions = list(region_activity.keys())
    if len(available_regions) < 2:
        continue

    # Labels for trials: 1 for context A, 2 for context B
    trial_labels = np.concatenate([np.ones(min_trials, dtype=int), np.full(min_trials, 2, dtype=int)])

    # Setup Cross-Validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Perform directed RRR predictive decoding for each ordered pair (source -> target)
    for source_region, target_region in permutations(available_regions, 2):
        pair_name = f'{source_region}->{target_region}'

        X_all = region_activity[source_region]  # (trials, bins, src_neurons)
        Y_all = region_activity[target_region]  # (trials, bins, tgt_neurons)

        rrr_acc_per_bin = np.zeros(n_bins)
        tgt_acc_per_bin = np.zeros(n_bins)
        src_acc_per_bin = np.zeros(n_bins)

        for b in range(n_bins):
            X_b = X_all[:, b, :]
            Y_b = Y_all[:, b, :]

            y_pred_rrr = np.zeros(total_trials)
            y_pred_tgt = np.zeros(total_trials)
            y_pred_src = np.zeros(total_trials)

            for train_idx, test_idx in skf.split(X_b, trial_labels):
                X_train, X_test = X_b[train_idx], X_b[test_idx]
                Y_train, Y_test = Y_b[train_idx], Y_b[test_idx]
                y_train, y_test = trial_labels[train_idx], trial_labels[test_idx]

                # Center source and target across training trials
                x_mean = np.mean(X_train, axis=0, keepdims=True)
                y_mean = np.mean(Y_train, axis=0, keepdims=True)

                X_train_c = X_train - x_mean
                Y_train_c = Y_train - y_mean
                X_test_c = X_test - x_mean

                # 1. Fit Reduced Rank Regression on training trials: Y_tgt = X_src * B
                B_rrr = fit_reduced_rank_regression(
                    X_train_c, Y_train_c, rank=RRR_RANK, ridge_alpha=RIDGE_ALPHA
                )

                # 2. Predict target activity for test trials using source activity
                Y_test_pred = (X_test_c @ B_rrr) + y_mean

                # 3. Train context decoder (LDA) on actual target activity Y_train
                lda_target = LinearDiscriminantAnalysis()
                lda_target.fit(Y_train, y_train)

                # Test context decoder on predicted target activity
                y_pred_rrr[test_idx] = lda_target.predict(Y_test_pred)

                # Test context decoder on actual target activity (benchmark)
                y_pred_tgt[test_idx] = lda_target.predict(Y_test)

                # 4. Context decoder directly on source region (for comparison)
                lda_source = LinearDiscriminantAnalysis()
                lda_source.fit(X_train, y_train)
                y_pred_src[test_idx] = lda_source.predict(X_test)

            rrr_acc_per_bin[b] = accuracy_score(trial_labels, y_pred_rrr)
            tgt_acc_per_bin[b] = accuracy_score(trial_labels, y_pred_tgt)
            src_acc_per_bin[b] = accuracy_score(trial_labels, y_pred_src)

            # Append results per bin
            for dec_type, acc in [('predicted_target_rrr', rrr_acc_per_bin[b]),
                                  ('actual_target', tgt_acc_per_bin[b]),
                                  ('source', src_acc_per_bin[b])]:
                results.append({
                    'subject': this_subject,
                    'session': date,
                    'source_region': source_region,
                    'target_region': target_region,
                    'region_pair': pair_name,
                    'position': unique_positions[b],
                    'accuracy': acc,
                    'decoder_type': dec_type,
                    'is_far': is_far
                })

results_df = pd.DataFrame(results)

# Save results to disk
save_dir = path_dict['save_path']
results_df.to_csv(save_dir / f'rrr_cross_region_decoding_{USE_TYPE}.csv', index=False)

# %% Plot Cross-Region Predictive Context Decoding for FAR Sessions
save_fig_dir = path_dict['paper_fig_path'] / 'RRR'
save_fig_dir.mkdir(parents=True, exist_ok=True)

for far_condition, cond_name, obj2_pos in [(1, 'FAR', 1325), (0, 'NEAR', 875)]:
    sub_df = results_df[(results_df['is_far'] == far_condition) &
                        (results_df['decoder_type'] == 'predicted_target_rrr')]
    unique_pairs = sorted(sub_df['region_pair'].unique())
    n_pairs = len(unique_pairs)

    if n_pairs == 0:
        continue

    n_cols = 5
    n_rows = int(np.ceil(n_pairs / n_cols))
    f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.5), dpi=dpi,
                          sharey=True, sharex=True)
    axs = np.array(axs).flatten()

    for idx, pair in enumerate(unique_pairs):
        ax = axs[idx]
        plot_df = sub_df[sub_df['region_pair'] == pair]

        sns.lineplot(data=plot_df, x='position', y='accuracy', ax=ax,
                     color='k', errorbar='se', err_kws={'lw': 0}, zorder=1)

        if len(plot_df['session'].unique()) >= 3:
            p_values, positions = run_stats_vs_chance(plot_df, chance_level=0.5)
            add_significance(positions, p_values, ax=ax, y_pos=0.88, alpha=0.05)

        ax.axhline(0.5, ls='--', lw=0.5, color='gray', zorder=0)
        ax.plot([425, 425], [0.3, 0.9], ls='--', lw=0.5, color='k', zorder=0)
        ax.plot([obj2_pos, obj2_pos], [0.3, 0.9], ls='--', lw=0.5, color='k', zorder=0)
        ax.set(title=pair, xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
               xlabel='', ylabel='', ylim=[0.3, 0.95], yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Hide unused axes
    for idx in range(n_pairs, len(axs)):
        axs[idx].axis('off')

    axs[0].set_ylabel('Predicted context accuracy', labelpad=0)
    f.supxlabel('Position (cm)', fontsize=7, y=0.04)
    f.suptitle(f'RRR Predicted Decoding ({cond_name})', fontsize=8)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(save_fig_dir / f'rrr_context_decoding_{cond_name.lower()}_{USE_TYPE}.pdf')
    plt.savefig(save_fig_dir / f'rrr_context_decoding_{cond_name.lower()}_{USE_TYPE}.jpg', dpi=600)
    plt.show()

