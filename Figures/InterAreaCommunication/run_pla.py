# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.cross_decomposition import PLSCanonical
from sklearn.model_selection import LeaveOneGroupOut
from joblib import Parallel, delayed
from scipy.stats import pearsonr, ranksums
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
TIME_WIN = {'obj1': [0, 2], 'obj2': [-2, 0], 'obj3': [-1, 1]}
BIN_SIZE = 0.025
SMOOTHING = 0.05
MIN_NEURONS = 5  # per region
N_CPUS = 18
N_COMPONENTS = 4

# Initialize
path_dict = paths()
subjects = load_subjects()

# Load in data
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv', dtype={'subject': str, 'date': str})
rec = rec.drop_duplicates(subset=['subject', 'date'])

# %% Fuction for parallization


# %%
corr_df = pd.DataFrame()

for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'Recording {i} of {len(rec)}: \n{subject} {date}')

    # Load in neural data for all probes
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
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

            n_trials, n_timebins, n_neurons_X = spikes_dict[obj][pair[0]].shape
            _, _, n_neurons_Y = spikes_dict[obj][pair[1]].shape

            # Initialize empty 3D arrays to store the latent trajectories
            # Shape will be: (Trials, Time, Components)
            X_latents = np.zeros((n_trials, n_timebins, N_COMPONENTS))
            Y_latents = np.zeros((n_trials, n_timebins, N_COMPONENTS))

            # Setup Leave-One-Trial-Out cross-validation
            logo = LeaveOneGroupOut()
            groups = np.arange(n_trials)

            # Initialize the calibrated PLS model
            pls = PLSCanonical(n_components=N_COMPONENTS)

            for train_idx, test_idx in logo.split(spikes_dict[obj][pair[0]], spikes_dict[obj][pair[1]], groups=groups):
                # test_idx is an array of length 1 containing the held-out trial number
                current_trial = test_idx[0]

                # Slice the 3D data into training and testing
                X_train_3D = spikes_dict[obj][pair[0]][train_idx]
                Y_train_3D = spikes_dict[obj][pair[1]][train_idx]

                X_test_3D = spikes_dict[obj][pair[0]][test_idx]
                Y_test_3D = spikes_dict[obj][pair[1]][test_idx]

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

            # Compute single trial coupling strength
            trial_coupling_scores = np.full(n_trials, np.nan)
            for trial in range(n_trials):

                component_correlations = np.full(N_COMPONENTS, np.nan)
                for comp in range(N_COMPONENTS):

                    # Calculate Pearson correlation
                    r_val, _ = pearsonr(X_latents[trial, :, comp], Y_latents[trial, :, comp])
                    component_correlations[comp] = np.abs(r_val)

                # Average the correlations across the 4 dimensions to get one score for the trial
                trial_coupling_scores[trial] = np.mean(component_correlations)







