# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from brainbox.population.decode import classify
from msvr_functions import paths

# Settings
MIN_TRIALS = 20
N_CORES = 18
N_NEURONS_SUB = 25  # Number of neurons to subselect
N_ITER = 50         # Number of iterations for subsampling

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(random_state=42, n_jobs=1, n_estimators=20, max_depth=5)

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position_20mms.pickle', 'rb') as handle:
    residuals_dict = pickle.load(handle)

# Change all cortex regions to 'Cortex'
targets = {'VIS', 'AUD', 'TEa', 'PERI', 'LEC'}
residuals_dict['region'] = [
    ['Cortex' if item in targets else item for item in sublist]
    for sublist in residuals_dict['region']]
residuals_dict['region'] = [np.array(i) for i in residuals_dict['region']]

# Function to run one full iteration (subsampling + decoding all bins)
def run_decoding_iteration(seed, X_full, y, positions, bin_centers, n_sub, classifier, cv):
    # Set seed for this specific worker to ensure reproducibility
    rng = np.random.default_rng(seed)
    
    # Subsample neurons
    n_total = X_full.shape[1]
    neuron_indices = rng.choice(n_total, n_sub, replace=False)
    X_sub = X_full[:, neuron_indices]
    
    # Initialize accuracy matrix (train_bin x test_bin)
    n_bins = len(bin_centers)
    acc_matrix = np.zeros((n_bins, n_bins))
    
    for i, train_bin in enumerate(bin_centers):
        mask_train = (positions == train_bin)
        X_train = X_sub[mask_train, :]
        y_train = y[mask_train]
        
        for j, test_bin in enumerate(bin_centers):
            if train_bin == test_bin:
                # Use cross-validation when training and testing on the same bin
                acc, _, _ = classify(X_train, y_train, classifier, cross_validation=cv)
                acc_matrix[i, j] = acc
            else:
                # Cross-decoding: Train on one bin, test on another
                mask_test = (positions == test_bin)
                X_test = X_sub[mask_test, :]
                y_test = y[mask_test]
                
                classifier.fit(X_train, y_train)
                acc_matrix[i, j] = classifier.score(X_test, y_test)
                
    return acc_matrix

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

        # Drop neurons with values smaller than float32 minimum
        X_decode_all = X_decode_all[:, ~np.any(X_decode_all < np.finfo(np.float32).min, axis=0)]

        # Get position bins
        rel_pos_bins = np.unique(residuals_dict['position'][i]).astype(int)
        
        # Run Parallel Iterations
        # We pass a unique seed (42 + n) to each worker
        print(f'Decoding {region}')
        results_list = Parallel(n_jobs=N_CORES)(
            delayed(run_decoding_iteration)(
                42 + n,                             # Seed
                X_decode_all,                       # Full neuron matrix
                residuals_dict['context'][i],       # Targets
                residuals_dict['position'][i].astype(int), # Position array
                rel_pos_bins,                       # Bins to loop over
                N_NEURONS_SUB,                      # N to subsample
                clf,                                # Classifier
                kfold_cv                            # CV Strategy
            )
            for n in range(N_ITER)
        )
        
        # results_list is a list of matrices (N_ITER x n_bins x n_bins)
        # Convert to array for easy averaging
        iter_results = np.array(results_list) 
        
        # Average accuracy over the iterations
        mean_accuracy = np.mean(iter_results, axis=0)
        
        # Create meshgrid for train/test positions
        train_pos, test_pos = np.meshgrid(rel_pos_bins, rel_pos_bins, indexing='ij')

        # Add to df
        decode_df = pd.concat((decode_df, pd.DataFrame(data={
            'accuracy': mean_accuracy.flatten(),
            'train_position': train_pos.flatten(),
            'test_position': test_pos.flatten(),
            'region': region,
            'n_neurons': N_NEURONS_SUB,
            'n_trials':  np.unique(residuals_dict['trial'][i]).shape[0],
            'subject': residuals_dict['subject'][i],
            'date': residuals_dict['date'][i],
            'probe': residuals_dict['probe'][i]
            })))    

# Save to disk
decode_df.to_csv(path_dict['save_path'] / 'decode_context_GLM_binpairs.csv', index=False)