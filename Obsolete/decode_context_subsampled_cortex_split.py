# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from msvr_functions import paths, classify

# Settings
MIN_TRIALS = 20
N_CORES = 20
MIN_STD = 0.05                  # Minimum variance of neurons to include them
N_NEURONS_SUB = 25              # Number of neurons to subselect
N_ITER = 50                     # Number of iterations for subsampling
DECODER = 'randomforest'        # regression, svm, randomforest, lda

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)

if DECODER == 'regression':
    clf = make_pipeline(RobustScaler(), LogisticRegression(random_state=42, max_iter=1000))
elif DECODER == 'randomforest':
    clf = RandomForestClassifier(random_state=42, n_jobs=1, n_estimators=20, max_depth=5)
elif DECODER == 'svm':
    clf = make_pipeline(RobustScaler(), SVC(random_state=42, max_iter=10000))
elif DECODER == 'lda':
    clf = LinearDiscriminantAnalysis()

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position_0mms.pickle', 'rb') as handle:
    residuals_dict = pickle.load(handle)

# Change all cortex regions to 'Cortex'
targets = {'VIS', 'AUD', 'TEa', 'PERI', 'LEC'}
residuals_dict['region'] = [
    ['Cortex' if item in targets else item for item in sublist]
    for sublist in residuals_dict['region']]
residuals_dict['region'] = [np.array(i) for i in residuals_dict['region']]

# Function to run one full iteration (subsampling + decoding all bins)
def run_decoding_iteration(seed, X_full, y, positions, bin_centers, n_sub, same_y, classifier, cv):
    # Set seed for this specific worker to ensure reproducibility
    rng = np.random.default_rng(seed)

    # Subsample neurons
    n_total = X_full.shape[1]
    neuron_indices = rng.choice(n_total, n_sub, replace=False)
    X_sub = X_full[:, neuron_indices]

    # Loop over position bins (Serial execution inside the worker)
    accuracies = []
    for bin_center in bin_centers:
        mask = (positions == bin_center)
        this_X = X_sub[mask, :]
        this_y = y[mask]

        # Select trials that are either the same or different context compared to the previous trial
        if same_y:
            use_trials = this_y[1:] == this_y[:-1]
        else:
            use_trials = this_y[1:] != this_y[:-1]
        this_X = this_X[1:, :][use_trials, :]
        this_y = this_y[1:][use_trials]

        # Decode
        acc, _ = classify(this_X, this_y, classifier, cross_validation=cv, return_prob=False)
        accuracies.append(acc)

    return accuracies


# Loop over recordings
decode_df = pd.DataFrame()
for i in range(len(residuals_dict['residuals'])):
    print(f'Recording {i} of {len(residuals_dict["residuals"])}')

    # Decode per brain region
    for r, region in enumerate(np.unique(residuals_dict['region'][i])):
        if region == 'root':
            continue

        # Select neurons from this brain region
        X_decode_all = residuals_dict['residuals'][i][:, residuals_dict['region'][i] == region]

        # Check constraints
        if X_decode_all.shape[1] < N_NEURONS_SUB:
            print('Too few neurons!')
            continue
        if np.unique(residuals_dict['trial'][i]).shape[0] < MIN_TRIALS:
            continue

        # Drop neurons with small variance
        X_decode_all = X_decode_all[:, np.std(X_decode_all, axis=0) >= MIN_STD]

        print(f'Decoding {region}')

        # Get position bins
        rel_pos_bins = np.unique(residuals_dict['position'][i]).astype(int)

        # Decode trials with the same context as the previous trial
        results_list = Parallel(n_jobs=N_CORES)(
            delayed(run_decoding_iteration)(
                42 + n,  # Seed
                X_decode_all,  # Full neuron matrix
                residuals_dict['context'][i],  # Targets
                residuals_dict['position'][i].astype(int),  # Position array
                rel_pos_bins,  # Bins to loop over
                N_NEURONS_SUB,  # N to subsample
                True,  # whether to include same or switch trials
                clf,  # Classifier
                kfold_cv  # CV Strategy
            )
            for n in range(N_ITER)
        )
        accuracy_same = np.mean(np.array(results_list), axis=0)

        # Decode trials with a different context as the previous trial
        results_list = Parallel(n_jobs=N_CORES)(
            delayed(run_decoding_iteration)(
                42 + n,  # Seed
                X_decode_all,  # Full neuron matrix
                residuals_dict['context'][i],  # Targets
                residuals_dict['position'][i].astype(int),  # Position array
                rel_pos_bins,  # Bins to loop over
                N_NEURONS_SUB,  # N to subsample
                False,  # whether to include same or switch trials
                clf,  # Classifier
                kfold_cv  # CV Strategy
            )
            for n in range(N_ITER)
        )
        accuracy_switch = np.mean(np.array(results_list), axis=0)

        # Add to df
        decode_df = pd.concat((decode_df, pd.DataFrame(data={
            'accuracy': np.concatenate((accuracy_same, accuracy_switch)),
            'position': np.concatenate((rel_pos_bins, rel_pos_bins)),
            'context': np.concatenate((['same'] * len(rel_pos_bins), ['switch'] * len(rel_pos_bins))),
            'region': region,
            'n_neurons': N_NEURONS_SUB,
            'n_trials': np.unique(residuals_dict['trial'][i]).shape[0],
            'subject': residuals_dict['subject'][i],
            'date': residuals_dict['date'][i],
            'probe': residuals_dict['probe'][i]
        })))

# Save to disk
decode_df.to_csv(path_dict['save_path'] / f'decode_context_cortex_{N_NEURONS_SUB}neurons_{DECODER}_same_switch.csv',
                 index=False)