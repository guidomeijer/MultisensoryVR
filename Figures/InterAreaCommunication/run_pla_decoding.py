# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSCanonical
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects, bin_signal

# Settings
TIME_WIN = {'obj1': [0, 2], 'obj2': [-2, 0]}
BIN_SIZE = 0.025
SMOOTHING = 0.05
MIN_NEURONS = 5  # per region
N_CPUS = -1
N_COMPONENTS = 4
N_SHUFFLES = 500
DECODING_BIN_SIZE = 0.3
DECODING_BIN_SHIFT = 0.05

# Initialize
path_dict = paths()
subjects = load_subjects()
clf = LogisticRegression(random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Load in data
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv', dtype={'subject': str, 'date': str})
rec = rec.drop_duplicates(subset=['subject', 'date'])

# Contruct bin centers for decoding
bin_centers = {'obj1': np.arange(TIME_WIN['obj1'][0] + (DECODING_BIN_SIZE / 2),
                                 TIME_WIN['obj1'][1] - (DECODING_BIN_SIZE / 2) + DECODING_BIN_SHIFT,
                                 step=DECODING_BIN_SHIFT),
               'obj2': np.arange(TIME_WIN['obj2'][0] + (DECODING_BIN_SIZE / 2),
                                 TIME_WIN['obj2'][1] - (DECODING_BIN_SIZE / 2) + DECODING_BIN_SHIFT,
                                 step=DECODING_BIN_SHIFT)}

# %% Fuctions

def fit_pla(x_region_a, x_region_b, use_n_components, do_shuffle=False):
    """
    Calculate the inter-regional coupling strength using PLS-Canonical and leave-one-trial-out cross-validation.

    Parameters
    ----------
    x_region_a, x_region_b : 3D array
        Neural data for two regions with shape (trials, timebins, neurons).
    use_n_components : int
        How many components to use for the PLS model
    do_shuffle : boolean
        Whether to shuffle trials of region b

    Returns
    -------
    trial_coupling_scores : 1D array
        The average Pearson correlation across latent components for each trial.
    X_latents, Y_latents : 3D array
        The latent trajectories for each trial and component.
        Shape is (trials, timebins, components).
    """

    n_trials, n_timebins, n_neurons_X = x_region_a.shape
    _, _, n_neurons_Y = x_region_b.shape

    # Do shuffle if required
    if do_shuffle:
        indices = np.arange(x_region_b.shape[0])
        shuffled_indices = np.random.permutation(x_region_b.shape[0])
        while np.any(shuffled_indices == indices):
            shuffled_indices = np.random.permutation(x_region_b.shape[0])
        x_region_b = x_region_b[shuffled_indices]

    # Initialize empty 3D arrays to store the latent trajectories
    # Shape will be: (Trials, Time, Components)
    X_latents = np.zeros((n_trials, n_timebins, use_n_components))
    Y_latents = np.zeros((n_trials, n_timebins, use_n_components))

    # Setup Leave-One-Trial-Out cross-validation
    logo = LeaveOneGroupOut()

    # Initialize the calibrated PLS model
    pls = PLSCanonical(n_components=use_n_components)

    for train_idx, test_idx in logo.split(x_region_a, x_region_b, groups=np.arange(n_trials)):
        # test_idx is an array of length 1 containing the held-out trial number
        current_trial = test_idx[0]

        # Slice the 3D data into training and testing
        X_train_3D = x_region_a[train_idx]
        Y_train_3D = x_region_b[train_idx]
        X_test_3D = x_region_a[test_idx]
        Y_test_3D = x_region_b[test_idx]

        # Flatten the training data to 2D for the PLS model: ((Trials * Time), Neurons)
        X_train_2D = X_train_3D.reshape(-1, n_neurons_X)
        Y_train_2D = Y_train_3D.reshape(-1, n_neurons_Y)

        # Flatten the single test trial to 2D: (Time, Neurons)
        X_test_2D = X_test_3D.reshape(-1, n_neurons_X)
        Y_test_2D = Y_test_3D.reshape(-1, n_neurons_Y)

        # Fit the model on the training trials
        pls.fit(X_train_2D, Y_train_2D)

        # Project the single unseen trial into the newly defined subspace
        X_latent_test, Y_latent_test = pls.transform(X_test_2D, Y_test_2D)

        # X_latent_test and Y_latent_test are shape (n_timebins, n_components).
        # We can store them directly into the pre-allocated 3D output arrays.
        X_latents[current_trial] = X_latent_test
        Y_latents[current_trial] = Y_latent_test

    return X_latents, Y_latents


def process_session(subject, date):
    """Process a single session and return a DataFrame with PLA results."""
    print(f'Processing session {subject} {date}...')
    session_pla_df = pd.DataFrame()

    # Load in neural data for all probes
    session_path = path_dict['local_data_path'] / 'Subjects' / str(subject) / str(date)
    spikes, clusters, channels = load_multiple_probes(session_path)

    # Load in object entry times
    all_obj_df = load_objects(subject, date)

    # Get binned spike counts per region
    spikes_dict = {'obj1': dict(), 'obj2': dict()}
    time_ax = dict()
    for k, probe in enumerate(spikes.keys()):
        for j, region in enumerate(np.unique(clusters[probe]['region'])):
            if region == 'root':
                continue
            
            # Select neurons
            region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]

            # Loop over objects
            for m, obj in enumerate(['obj1', 'obj2']):
        
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

                # Get time axis
                time_ax[obj] = peth['tscale']

    # Loop over objects
    for m, obj in enumerate(['obj1', 'obj2']):

        # Sort pairs alphabetically so that there are never any duplicates
        region_pairs = sorted([tuple(sorted(pair)) for pair in list(combinations(spikes_dict[obj].keys(), 2))])

        # Loop over region pairs
        for pair in region_pairs:

            # Get latents
            latents_region1, latents_region2 = fit_pla(
                spikes_dict[obj][pair[0]], spikes_dict[obj][pair[1]], use_n_components=N_COMPONENTS)

            # Bin latents for decoding
            decode_region1 = np.full((latents_region1.shape[0], bin_centers[obj].shape[0], latents_region1.shape[2]), np.nan)
            for kk in range(latents_region1.shape[2]):
                for tt in range(latents_region1.shape[0]):
                    decode_region1[tt, :, kk] = bin_signal(
                        time_ax[obj], latents_region1[tt, :, kk], bin_centers[obj], DECODING_BIN_SIZE)
            decode_region2 = np.full((latents_region2.shape[0], bin_centers[obj].shape[0], latents_region2.shape[2]), np.nan)
            for kk in range(latents_region2.shape[2]):
                for tt in range(latents_region2.shape[0]):
                    decode_region2[tt, :, kk] = bin_signal(
                        time_ax[obj], latents_region2[tt, :, kk], bin_centers[obj], DECODING_BIN_SIZE)

            # Reshape into (trials x latents x timebins)
            decode_region1 = np.swapaxes(decode_region1, 1, 2)
            decode_region2 = np.swapaxes(decode_region2, 1, 2)

            # Perform decoding per timebin
            trial_labels = all_obj_df.loc[all_obj_df['object'] == m+1, 'goal'].values
            accuracy_region1 = np.full(decode_region1.shape[2], np.nan)
            accuracy_region2 = np.full(decode_region1.shape[2], np.nan)
            for tt in range(decode_region1.shape[2]):

                # Decode region 1
                pred = np.empty(decode_region1.shape[0])
                for train_index, test_index in cv.split(decode_region1[:, :, tt], trial_labels):

                    # Fit to training data and test on held out test data
                    clf.fit(decode_region1[:, :, tt][train_index], trial_labels[train_index])
                    pred[test_index] = clf.predict(decode_region1[:, :, tt][test_index])

                # Get accuracy
                accuracy_region1[tt] = accuracy_score(trial_labels, pred)

                # Decode region 2
                pred = np.empty(decode_region2.shape[0])
                for train_index, test_index in cv.split(decode_region2[:, :, tt], trial_labels):
                    # Fit to training data and test on held out test data
                    clf.fit(decode_region2[:, :, tt][train_index], trial_labels[train_index])
                    pred[test_index] = clf.predict(decode_region2[:, :, tt][test_index])

                # Get accuracy
                accuracy_region2[tt] = accuracy_score(trial_labels, pred)

            # Add to dataframe
            session_pla_df = pd.concat([session_pla_df, pd.DataFrame(data={
                'accuracy': np.concatenate((accuracy_region1, accuracy_region2)),
                'time': np.concatenate((bin_centers[obj], bin_centers[obj])),
                'region': np.concatenate((np.full(accuracy_region1.shape, pair[0]),
                                          np.full(accuracy_region2.shape, pair[1]))),
                'object': obj, 'region_pair': f'{pair[0]}-{pair[1]}',
                'subject': subject, 'date': date})])

    return session_pla_df


# %% Execute parallel processing
if __name__ == '__main__':
    print(f'Starting parallel processing of {len(rec)} sessions using {N_CPUS} CPUs...')
    
    # Run parallel loop over sessions
    results = Parallel(n_jobs=N_CPUS)(
        delayed(process_session)(row['subject'], row['date']) 
        for _, row in rec.iterrows()
    )

    # Concatenate all session DataFrames into one
    pla_df = pd.concat(results, ignore_index=True)

    # Save to disk
    pla_df.to_csv(path_dict['google_drive_data_path'] / 'pla_decoding.csv', index=False)
    
    print('Processing complete.')
