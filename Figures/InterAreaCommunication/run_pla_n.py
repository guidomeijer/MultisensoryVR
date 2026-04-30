# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.cross_decomposition import PLSCanonical
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from scipy.stats import pearsonr, sem
import warnings
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
T_BEFORE = 0
T_AFTER = 2
OBJ = 1
BIN_SIZE = 0.025
SMOOTHING = 0.05
MIN_NEURONS = 10  # per region
N_CPUS = 18
MAX_DIM = 10

# Initialize
path_dict = paths()
subjects = load_subjects()
gkf = GroupKFold(n_splits=10)

# Load in data
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv', dtype={'subject': str, 'date': str})
rec = rec.drop_duplicates(subset=['subject', 'date'])

def process_session(subject, date, path_dict):
    print(f'Recording: {subject} {date}')
    session_pla_df = pd.DataFrame()

    # Load in neural data for all probes
    session_path = path_dict['local_data_path'] / 'Subjects' / str(subject) / str(date)
    spikes, clusters, channels = load_multiple_probes(session_path)

    # Load in object entry times
    all_obj_df = load_objects(subject, date)
    trials = pd.read_csv(path_dict['local_data_path'] / 'Subjects' / subject / date / 'trials.csv')

    # Get binned spike counts per region
    spikes_dict = dict()
    trial_ids = None
    for probe in spikes.keys():
        for j, region in enumerate(np.unique(clusters[probe]['region'])):
            if region == 'root':
                continue
            
            # Select neurons
            region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]

            # Get spike counts (trials x neurons x time)
            peth, binned_spikes = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], region_neurons,
                all_obj_df.loc[all_obj_df['object'] == OBJ, 'times'].values,
                T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING, return_fr=False)

            # Reshape into (trials x time x neurons)
            spikes_transposed = np.swapaxes(binned_spikes, 1, 2)

            # Drop silent neurons
            spikes_dropped = spikes_transposed[:, :, np.max(peth['means'], axis=1) > 0.01]
            if spikes_dropped.shape[2] < MIN_NEURONS:
                continue

            # Subtract mean over trials from each trial to leave the residuals
            spikes_residuals =  spikes_dropped - np.mean(spikes_dropped, axis=0)

            # Restructure binned_spikes into time x neurons
            spikes_dict[region] = spikes_residuals.reshape(-1, spikes_residuals.shape[2])
            if trial_ids is None:
                trial_ids = np.repeat(np.arange(binned_spikes.shape[0]), binned_spikes.shape[2])

    if len(spikes_dict) < 2:
        return session_pla_df

    # Sort pairs alphabetically so that there are never any duplicates
    region_pairs = sorted([tuple(sorted(pair)) for pair in list(combinations(spikes_dict.keys(), 2))])

    # Loop over region pairs
    for pair in region_pairs:
        
        # Loop over number of components
        cv_scores = np.full(MAX_DIM, np.nan)
        for n in range(MAX_DIM):
            pls = PLSCanonical(n_components=n+1)
            fold_scores = []

            for train_idx, test_idx in gkf.split(spikes_dict[pair[0]], spikes_dict[pair[1]], groups=trial_ids):

                X_train, Y_train = spikes_dict[pair[0]][train_idx], spikes_dict[pair[1]][train_idx]
                X_test, Y_test = spikes_dict[pair[0]][test_idx], spikes_dict[pair[1]][test_idx]

                pls.fit(X_train, Y_train)

                # Predict the full matrix using ALL n components
                Y_pred = pls.predict(X_test)

                # Evaluate how well the model predicts the temporal dynamics of each neuron
                neuron_corrs = []
                n_neurons = Y_test.shape[1]

                for neuron_idx in range(n_neurons):

                    # Temporarily suppress warnings for this specific calculation
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r_val, _ = pearsonr(Y_test[:, neuron_idx], Y_pred[:, neuron_idx])

                    # Ignore neurons that were completely silent in the test set (variance = 0)
                    if not np.isnan(r_val):
                        neuron_corrs.append(r_val)

                # The fold score is the average prediction correlation across the population
                fold_scores.append(np.mean(neuron_corrs))

            # Average the scores across the folds
            cv_scores[n] = np.mean(fold_scores)

        # Normalize
        cv_scores_norm = cv_scores / np.max(cv_scores)

        # Add to session dataframe
        session_pla_df = pd.concat((session_pla_df, pd.DataFrame(data={
            'r': cv_scores, 'r_norm': cv_scores_norm, 'n_components': np.arange(1, MAX_DIM+1),
            'region_pair': f'{pair[0]}-{pair[1]}', 'region_1': pair[0], 'region_2': pair[1],
            'subject': subject, 'date': date})))
            
    return session_pla_df

# %% Run parallel processing
results = Parallel(n_jobs=N_CPUS)(
    delayed(process_session)(row['subject'], row['date'], path_dict) 
    for _, row in rec.iterrows()
)

# Concatenate and save to disk
pla_df = pd.concat(results, ignore_index=True)
pla_df.to_csv(path_dict['google_drive_data_path'] / 'pla_n_components.csv', index=False)
print('Done!')
