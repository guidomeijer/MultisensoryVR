# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
from os.path import join
import pandas as pd
import pickle
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from msvr_functions import paths, load_subjects

def shuffle_iteration(X_train, y_train, X_bins, clf):
    shuf_y = shuffle(y_train)
    clf.fit(X_train, shuf_y)
    dists = []
    for X_test in X_bins:
        lda_proj = clf.transform(X_test)
        dists.append(np.abs(np.mean(lda_proj[shuf_y == 1]) - np.mean(lda_proj[shuf_y == 2])))
    return dists

# Settings
MIN_NEURONS = 10
MIN_TRIALS = 20
LDA_POS = 600
N_SHUFFLE = 500
N_CPUS = 18

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
subjects = load_subjects()
lda_dist_df = pd.DataFrame()
clf_lda = LinearDiscriminantAnalysis()
scaler = StandardScaler()
selector = VarianceThreshold(threshold=0)

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position.pickle', 'rb') as handle:
    residuals_dict = pickle.load(handle)

for i in range(len(residuals_dict['residuals'])):
    print(f'Recording {i} of {len(residuals_dict["residuals"])}')
    
    # Fit LDA projection per region
    for r, region in enumerate(np.unique(residuals_dict['region'][i])):
        if region == 'root':
            continue
        
        # Select neurons from this brain region
        X_raw = residuals_dict['residuals'][i][:, residuals_dict['region'][i] == region]
        
        # Check constraints
        if X_raw.shape[1] < MIN_NEURONS: continue
        if np.unique(residuals_dict['trial'][i]).shape[0] < MIN_TRIALS: continue
    
        # Fit scaler and selector on the ENTIRE dataset (all positions)
        try:
            X_global_selected = selector.fit_transform(X_raw)
            X_global_scaled = scaler.fit_transform(X_global_selected)
        except ValueError:
            # Handle cases where all features might be variance 0
            continue
                
        # Get position bin centers
        rel_pos_bins = np.unique(residuals_dict['position'][i])
        
        # DEFINE TRAINING SET (At specific location)
        # Find index of data points at LDA_POS
        proj_bin = rel_pos_bins[np.argmin(np.abs(rel_pos_bins - LDA_POS))]
        train_mask = residuals_dict['position'][i] == proj_bin
        
        X_train = X_global_scaled[train_mask, :]
        y_train = residuals_dict['context'][i][train_mask]
        
        # FIT LDA (Locally)
        clf_lda.fit(X_train, y_train)
        
        # Loop over bins
        lda_dist = np.empty(rel_pos_bins.shape[0])
        X_bins = [] # If you need this for shuffling, store the SCALED chunks
        
        for j, bin_center in enumerate(rel_pos_bins):
            # Extract the globally scaled data for this bin
            bin_mask = residuals_dict['position'][i] == bin_center
            this_X = X_global_scaled[bin_mask, :]
            this_y = residuals_dict['context'][i][bin_mask]
            
            # Save for shuffle
            X_bins.append(this_X)
            
            # Project using the LDA weights (no re-scaling here)
            lda_proj = clf_lda.transform(this_X)
            
            # Calculate distance
            # Check if both classes exist in this bin to avoid errors
            if len(np.unique(this_y)) == 2:
                lda_dist[j] = np.abs(np.mean(lda_proj[this_y == 1]) - np.mean(lda_proj[this_y == 2]))
            else:
                lda_dist[j] = np.nan # Or 0, depending on preference
        asd
        # Do the same for the shuffles
        shuf_dist = Parallel(n_jobs=N_CPUS)(delayed(shuffle_iteration)(
                                        X_train, y_train, X_bins, clf_lda) for k in range(N_SHUFFLE))
        shuf_dist = np.array(shuf_dist)

        # Get 95% confidence interval of shuffled distribution                 
        ci_bounds = np.quantile(shuf_dist, [0.025, 0.975], axis=0)
        lower_bound = ci_bounds[0]
        upper_bound = ci_bounds[1]
        
        # Add to dataframe
        lda_dist_df = pd.concat((lda_dist_df, pd.DataFrame(data={
            'position': rel_pos_bins, 'lda_distance': lda_dist,
            'lda_shuf_lower': lower_bound, 'lda_shuf_upper': upper_bound,
            'region': region,
            'subject': residuals_dict['subject'][i],
            'date': residuals_dict['date'][i],
            'probe': residuals_dict['probe'][i]})))
    
    # Save to disk
    lda_dist_df.to_csv(join(path_dict['save_path'], 'lda_distance_context.csv'), index=False)
            
            
            
            
        
    