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
from sklearn.utils import shuffle
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
PREV_TRIAL = False               # Whether to decode the context of the previous trial
SHUFFLE = True                  # Whether to shuffle trial labels

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
def run_decoding_iteration(seed, X_full, y, positions, bin_centers, n_sub, prev_trial, do_shuffle, classifier, cv):
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

        # shift trials if necessary
        if prev_trial:
            this_X = this_X[1:, :]
            this_y = this_y[:-1]

        # shuffle trials if necessary
        if do_shuffle:
            this_y = shuffle(this_y)

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

        # Run Parallel Iterations
        # We pass a unique seed (42 + n) to each worker
        results_list = Parallel(n_jobs=N_CORES)(
            delayed(run_decoding_iteration)(
                42 + n,  # Seed
                X_decode_all,  # Full neuron matrix
                residuals_dict['context'][i],  # Targets
                residuals_dict['position'][i].astype(int),  # Position array
                rel_pos_bins,  # Bins to loop over
                N_NEURONS_SUB,  # N to subsample
                PREV_TRIAL,  # whether to decode current or previous trial
                SHUFFLE,     # whether to shuffle trial labels
                clf,  # Classifier
                kfold_cv  # CV Strategy
            )
            for n in range(N_ITER)
        )

        # results_list is a list of lists (100 iterations x N bins)
        # Convert to array for easy averaging
        iter_results = np.array(results_list)

        # Average accuracy over the 100 iterationscontext_df
        mean_accuracy = np.mean(iter_results, axis=0)

        # Add to df
        decode_df = pd.concat((decode_df, pd.DataFrame(data={
            'accuracy': mean_accuracy,
            'position': rel_pos_bins,
            'region': region,
            'n_neurons': N_NEURONS_SUB,
            'n_trials': np.unique(residuals_dict['trial'][i]).shape[0],
            'subject': residuals_dict['subject'][i],
            'date': residuals_dict['date'][i],
            'probe': residuals_dict['probe'][i]
        })))

# Save to disk
append_str = ''
if PREV_TRIAL:
    append_str = append_str + '_prevtrial'
if SHUFFLE:
    append_str = append_str + '_shuffle'
decode_df.to_csv(path_dict['save_path'] / f'decode_context_cortex_{N_NEURONS_SUB}neurons_{DECODER}{append_str}.csv',
                 index=False)