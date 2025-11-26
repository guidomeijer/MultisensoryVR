# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from brainbox.population.decode import classify
from msvr_functions import paths

# Settings
MIN_NEURONS = 10
MIN_TRIALS = 20
N_CORES = 12

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
kfold_cv = KFold(n_splits=10, shuffle=True, random_state=42)
clf = RandomForestClassifier(random_state=42, n_jobs=1)
scaler = StandardScaler()

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position.pickle', 'rb') as handle:
    residuals_dict = pickle.load(handle)

# Function for parallel processing
def decode_context(bin_center, X_decode, y, X_bin_centers):

    this_X = X_decode[X_bin_centers == bin_center, :]
    this_y = y[X_bin_centers == bin_center]
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

        # Do decoding per position bin
        rel_pos_bins = np.unique(residuals_dict['position'][i]).astype(int)
        results = Parallel(n_jobs=N_CORES)(
            delayed(decode_context)(bin_center,
                                    X_decode,
                                    residuals_dict['context'][i],
                                    residuals_dict['position'][i].astype(int))
            for bin_center in rel_pos_bins)
        accuracy = [i for i in results]

        # Add to df
        decode_df = pd.concat((decode_df, pd.DataFrame(data={
            'accuracy': accuracy,
            'position': rel_pos_bins,
            'region': region,
            'subject': residuals_dict['subject'][i],
            'date': residuals_dict['date'][i],
            'probe': residuals_dict['probe'][i]
            })))   

# Save to disk
decode_df.to_csv(path_dict['save_path'] / 'decode_context_GLM_position.csv', index=False)
