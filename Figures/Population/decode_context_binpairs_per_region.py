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
from sklearn.base import clone
from joblib import Parallel, delayed
from msvr_functions import paths

# Settings
MIN_TRIALS = 10
N_CORES = 18
DECODER = 'randomforest'      # regression, svm, randomforest, lda

# Initialize
path_dict = paths(sync=False)
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

def decode_single_train_bin(train_bin, rel_pos_bins, X_decode_all, y, positions, trials, train_trials, test_trials, classifier):
    train_mask = (positions == train_bin) & np.isin(trials, train_trials)
    X_train = X_decode_all[train_mask]
    y_train = y[train_mask]
    
    row = np.zeros(len(rel_pos_bins))
    if len(np.unique(y_train)) < 2:
        row[:] = np.nan
        return row
        
    local_clf = clone(classifier)
    local_clf.fit(X_train, y_train)
    
    for j_bin, test_bin in enumerate(rel_pos_bins):
        test_mask = (positions == test_bin) & np.isin(trials, test_trials)
        X_test = X_decode_all[test_mask]
        y_test = y[test_mask]
        
        if len(X_test) == 0:
            row[j_bin] = np.nan
        else:
            row[j_bin] = local_clf.score(X_test, y_test)
            
    return row

# Loop over recordings
decode_df = pd.DataFrame()
for i in range(len(residuals_dict['residuals'])):
    print(f'Recording {i+1} of {len(residuals_dict["residuals"])}')
    
    unique_trials = np.unique(residuals_dict['trial'][i])
    if unique_trials.shape[0] < MIN_TRIALS:
        continue

    # Decode per brain region
    for r, region in enumerate(np.unique(residuals_dict['region'][i])):
        if region == 'root':
            continue
        
        # Select neurons from this brain region
        X_decode_all = residuals_dict['residuals'][i][:, residuals_dict['region'][i] == region]

        # Check constraints
        if X_decode_all.shape[1] == 0:
            continue

        # Drop neurons with values smaller than float32 minimum
        X_decode_all = X_decode_all[:, ~np.any(X_decode_all < np.finfo(np.float32).min, axis=0)]
        
        n_neurons = X_decode_all.shape[1]
        if n_neurons == 0:
            continue

        # Get position bins
        rel_pos_bins = np.unique(residuals_dict['position'][i]).astype(int)
        n_bins = len(rel_pos_bins)
        
        print(f'Decoding {region} (N={n_neurons} neurons)')
        
        acc_matrix_folds = np.zeros((kfold_cv.get_n_splits(), n_bins, n_bins))
        
        for fold, (train_trial_idx, test_trial_idx) in enumerate(kfold_cv.split(unique_trials)):
            train_trials = unique_trials[train_trial_idx]
            test_trials = unique_trials[test_trial_idx]
            
            results = Parallel(n_jobs=N_CORES)(
                delayed(decode_single_train_bin)(
                    train_bin, rel_pos_bins, X_decode_all, 
                    residuals_dict['context'][i], residuals_dict['position'][i].astype(int), 
                    residuals_dict['trial'][i], train_trials, test_trials, clf
                )
                for train_bin in rel_pos_bins
            )
            
            acc_matrix_folds[fold, :, :] = np.array(results)
        
        mean_accuracy = np.nanmean(acc_matrix_folds, axis=0)
        
        # Create meshgrid for train/test positions
        train_pos, test_pos = np.meshgrid(rel_pos_bins, rel_pos_bins, indexing='ij')

        # Add to df
        decode_df = pd.concat((decode_df, pd.DataFrame(data={
            'accuracy': mean_accuracy.flatten(),
            'train_position': train_pos.flatten(),
            'test_position': test_pos.flatten(),
            'region': region,
            'n_neurons': n_neurons,
            'n_trials':  unique_trials.shape[0],
            'subject': residuals_dict['subject'][i],
            'date': residuals_dict['date'][i],
            'probe': residuals_dict['probe'][i]
            })))    

# Save to disk
decode_df.to_csv(path_dict['save_path'] / f'decode_context_binpairs_{DECODER}.csv', index=False)
