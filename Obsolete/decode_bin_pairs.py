# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 12/03/2026
"""
# %%

import numpy as np
import pandas as pd
import pickle
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from brainbox.population.decode import classify
from msvr_functions import paths

# Settings
MIN_NEURONS = 2
MIN_TRIALS = 10
N_CORES = 18

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(random_state=42, n_jobs=1, n_estimators=100, max_depth=5)
# clf = LinearDiscriminantAnalysis()
scaler = StandardScaler()

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position.pickle', 'rb') as handle:
    residuals_dict = pickle.load(handle)


# Function for parallel processing
def decode_pair(bin1, bin2, X_decode, X_bin_centers):
    this_X = X_decode[(X_bin_centers == bin1) | (X_bin_centers == bin2), :]
    this_y = X_bin_centers[(X_bin_centers == bin1) | (X_bin_centers == bin2)]
    accuracy, _, _ = classify(this_X, this_y, clf, cross_validation=kfold_cv)

    return accuracy


# Loop over recordings
decode_df = pd.DataFrame()
for i in range(len(residuals_dict['residuals'])):
    print(f'Recording {i} of {len(residuals_dict["residuals"])}')

    # Decode per brain region
    for r, region in enumerate(np.unique(residuals_dict['region'][i])):
        if region == 'root':
            continue
        print(f'Decoding {region}')

        # Select neurons from this brain region
        X_decode = residuals_dict['residuals'][i][:, residuals_dict['region'][i] == region]

        # Skip region if
        if X_decode.shape[1] < MIN_NEURONS:
            continue
        if np.unique(residuals_dict['trial'][i]).shape[0] < MIN_TRIALS:
            continue

        # Do decoding per position bin pair
        rel_pos_bins = np.unique(residuals_dict['position'][i]).astype(int)
        bin_pairs = list(combinations(rel_pos_bins, 2))
        results = Parallel(n_jobs=N_CORES)(
            delayed(decode_pair)(bin1, bin2, X_decode, residuals_dict['position'][i].astype(int))
            for bin1, bin2 in bin_pairs)

        # Populate 2D array
        n_bins = len(rel_pos_bins)
        decode_matrix = np.full((n_bins, n_bins), np.nan)
        bin_map = {b: k for k, b in enumerate(rel_pos_bins)}
        for (b1, b2), acc in zip(bin_pairs, results):
            decode_matrix[bin_map[b1], bin_map[b2]] = acc
            decode_matrix[bin_map[b2], bin_map[b1]] = acc

        # Add to df
        decode_df = pd.concat((decode_df, pd.DataFrame(data={
            'region': region,
            'n_neurons': X_decode.shape[1],
            'n_trials': np.unique(residuals_dict['trial'][i]).shape[0],
            'subject': residuals_dict['subject'][i],
            'date': residuals_dict['date'][i],
            'probe': residuals_dict['probe'][i],
            'bins': [rel_pos_bins],
            'decode_matrix': [decode_matrix]
        })), ignore_index=True)

# Save to disk
decode_df.to_pickle(path_dict['save_path'] / 'decode_bin_pairs.pkl')