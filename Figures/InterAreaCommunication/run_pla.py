# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.cross_decomposition import PLSCanonical
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
TIME_WIN = {'obj1': [0, 2], 'obj2': [-2, 0], 'obj3': [-1, 1]}
BIN_SIZE = 0.025
SMOOTHING = 0.05
MIN_NEURONS = 5  # per region
N_CPUS = -6
N_COMPONENTS = 4
N_SHUFFLES = 500
N_SPLITS = 10

# Initialize
path_dict = paths()
subjects = load_subjects()

# Load in data
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv', dtype={'subject': str, 'date': str})
rec = rec.drop_duplicates(subset=['subject', 'date'])

# %% Fuctions
def fit_pla(x_region_a, x_region_b, use_n_components, do_shuffle=False, n_splits=10):
    n_trials, n_timebins, n_neurons_X = x_region_a.shape
    _, _, n_neurons_Y = x_region_b.shape

    # Fast shuffle
    if do_shuffle:
        x_region_b = x_region_b[np.random.permutation(n_trials)]

    X_latents = np.zeros((n_trials, n_timebins, use_n_components))
    Y_latents = np.zeros((n_trials, n_timebins, use_n_components))

    # 5-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True)

    # Initialize PLS
    pls = PLSCanonical(n_components=use_n_components)

    for train_idx, test_idx in kf.split(np.arange(n_trials)):
        # 1. Slice
        X_train_3D, Y_train_3D = x_region_a[train_idx], x_region_b[train_idx]
        X_test_3D, Y_test_3D = x_region_a[test_idx], x_region_b[test_idx]

        # 2. Flatten
        X_train_2D = X_train_3D.reshape(-1, n_neurons_X)
        Y_train_2D = Y_train_3D.reshape(-1, n_neurons_Y)
        X_test_2D = X_test_3D.reshape(-1, n_neurons_X)
        Y_test_2D = Y_test_3D.reshape(-1, n_neurons_Y)

        # 3. Apply PCA to keep components explaining 90% of the variance
        # This prevents PLS from fitting to single-neuron noise
        pca_X = PCA(n_components=0.90)
        pca_Y = PCA(n_components=0.90)

        X_train_pca = pca_X.fit_transform(X_train_2D)
        Y_train_pca = pca_Y.fit_transform(Y_train_2D)

        # Transform test set using the learned PCA space
        X_test_pca = pca_X.transform(X_test_2D)
        Y_test_pca = pca_Y.transform(Y_test_2D)

        # 4. Fit PLS on the reduced PCA space
        actual_components = min(use_n_components, X_train_pca.shape[1], Y_train_pca.shape[1])
        pls.set_params(n_components=actual_components)

        pls.fit(X_train_pca, Y_train_pca)
        X_latent_test, Y_latent_test = pls.transform(X_test_pca, Y_test_pca)

        # 5. Reshape and store
        n_test_trials = len(test_idx)
        X_latents[test_idx, :, :actual_components] = X_latent_test.reshape(n_test_trials, n_timebins, actual_components)
        Y_latents[test_idx, :, :actual_components] = Y_latent_test.reshape(n_test_trials, n_timebins, actual_components)

    # Compute trial coupling scores
    trial_coupling_scores = np.full(n_trials, np.nan)
    for trial in range(n_trials):
        component_correlations = np.full(actual_components, np.nan)
        for comp in range(actual_components):
            # Check for zero variance to avoid pearsonr warnings
            if np.std(X_latents[trial, :, comp]) > 0 and np.std(Y_latents[trial, :, comp]) > 0:
                r_val, _ = pearsonr(X_latents[trial, :, comp], Y_latents[trial, :, comp])
                component_correlations[comp] = np.abs(r_val)
            else:
                component_correlations[comp] = 0.0

        trial_coupling_scores[trial] = np.nanmean(component_correlations)

    return trial_coupling_scores

pla_df = pd.DataFrame()
for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'Processing session {i} of {rec.shape[0]} ({subject} {date})')

    # Load in neural data for all probes
    session_path = path_dict['local_data_path'] / 'Subjects' / str(subject) / str(date)
    spikes, clusters, channels = load_multiple_probes(session_path)

    # Load in object entry times
    all_obj_df = load_objects(subject, date)
    trials = pd.read_csv(path_dict['local_data_path'] / 'Subjects' / subject / date / 'trials.csv')

    # Get binned spike counts per region
    spikes_dict = {'obj1': dict(), 'obj2': dict(), 'obj3': dict()}
    trial_ids = None
    for k, probe in enumerate(spikes.keys()):
        for j, region in enumerate(np.unique(clusters[probe]['region'])):
            if region == 'root':
                continue

            # Select neurons
            region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]

            # Loop over objects
            for m, obj in enumerate(['obj1', 'obj2', 'obj3']):

                # Get spike counts (trials x neurons x time)
                peth, binned_spikes = calculate_peths(
                    spikes[probe]['times'], spikes[probe]['clusters'], region_neurons,
                    all_obj_df.loc[all_obj_df['object'] == m+1, 'times'].values,
                    np.abs(TIME_WIN[obj][0]), TIME_WIN[obj][1], BIN_SIZE, SMOOTHING, return_fr=False)

                # Reshape into (trials x time x neurons)
                spikes_transposed = np.swapaxes(binned_spikes, 1, 2)

                # Drop silent neurons
                spikes_dropped = spikes_transposed[:, :, np.max(peth['means'], axis=1) > 0.01]
                if spikes_dropped.shape[2] < MIN_NEURONS:
                    continue

                # Subtract the mean over trials from each trial to leave the residuals
                spikes_dict[obj][region] = spikes_dropped - np.mean(spikes_dropped, axis=0)

    # Loop over objects
    for m, obj in enumerate(['obj1', 'obj2', 'obj3']):

        # Sort pairs alphabetically so that there are never any duplicates
        region_pairs = sorted([tuple(sorted(pair)) for pair in list(combinations(spikes_dict[obj].keys(), 2))])

        # Loop over region pairs
        for pair in region_pairs:
            print(f'{pair[0]} - {pair[1]}')

            # Calculate true coupling
            coupling_scores = fit_pla(
                spikes_dict[obj][pair[0]],
                spikes_dict[obj][pair[1]],
                use_n_components=N_COMPONENTS,
                n_splits=N_SPLITS
            )
            mean_coupling = np.mean(coupling_scores)

            # Define a wrapper for the parallel shuffle
            def get_shuffled_mean(X, Y, n_comp):
                scores = fit_pla(X, Y, use_n_components=n_comp, do_shuffle=True, n_splits=N_SPLITS)
                return np.mean(scores)

            # Execute shuffles in parallel
            mean_shuffled = Parallel(n_jobs=N_CPUS)(
                delayed(get_shuffled_mean)(
                    spikes_dict[obj][pair[0]],
                    spikes_dict[obj][pair[1]],
                    N_COMPONENTS
                ) for _ in range(N_SHUFFLES)
            )

            # Calculate p-value
            mean_shuffled = np.array(mean_shuffled)
            p_coupling = np.sum(mean_shuffled >= mean_coupling) / N_SHUFFLES

            # Add to dataframe
            pla_df = pd.concat([pla_df, pd.DataFrame(index=[pla_df.shape[0]], data={
                'coupling_r': mean_coupling, 'coupling_p': p_coupling,
                'object': obj, 'region_pair': f'{pair[0]}-{pair[1]}', 'region_1': pair[0], 'region_2': pair[1],
                'subject': subject, 'date': date})])

    # Save to disk
    pla_df.to_csv(path_dict['google_drive_data_path'] / 'pla.csv', index=False)

