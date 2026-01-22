# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from spikeship import spikeship
from joblib import Parallel, delayed
from msvr_functions import (paths, load_neural_data, load_subjects, load_objects,
                            to_spikeship_dataformat)

# Settings
T_BEFORE = 2  # s
T_AFTER = 2
BIN_SIZE = 0.3
STEP_SIZE = 0.05
MIN_NEURONS = 3
MIN_TRIALS = 20
N_CPUS = 18
N_FOLDS = 10
GAMMA_SCALE = 1

# Create time array
t_centers = np.arange(-T_BEFORE + (BIN_SIZE/2), T_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# Functions
def run_spikeship(this_bin_center, use_spikes, use_clusters, event_times):
        
    # Get time intervals
    these_intervals = np.vstack((event_times + (this_bin_center - (BIN_SIZE/2)),
                                 event_times + (this_bin_center + (BIN_SIZE/2)))).T
    
    # Transform to spikeship data format
    ss_spike_times, ii_spike_times, n_spikes = to_spikeship_dataformat(
        use_spikes, use_clusters, these_intervals)
    
    # Run SpikeShip
    diss_spikeship = spikeship.distances(ss_spike_times, ii_spike_times)
    
    return diss_spikeship

def decode_spikeship_knn(diss_arr, trial_labels, n_neighbors=5):
    """
    Decodes conditions using k-NN on a SpikeShip dissimilarity matrix.
    """
    # 1. Validation and Symmetry check
    if diss_arr.shape[0] != len(trial_labels):
        raise ValueError(f"Shape mismatch: Matrix {diss_arr.shape}, labels {len(trial_labels)}")
    
    # Ensure symmetry: D = (D + D.T) / 2
    # This prevents issues with certain 'precomputed' algorithms
    D = (diss_arr + diss_arr.T) / 2.0

    # 2. Setup Cross-Validation
    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    accuracies = []
    trial_indices = np.arange(len(trial_labels))

    for train_idx, test_idx in skf.split(trial_indices):
        y_train = trial_labels[train_idx]
        y_test = trial_labels[test_idx]
        
        # 3. Slice the distance matrix
        # k-NN 'precomputed' expects (n_queries, n_indexed)
        # i.e., distance from each test trial to each training trial
        D_train_test = D[np.ix_(test_idx, train_idx)]
        
        # 4. Initialize and fit k-NN
        # 'brute' is required for precomputed distances
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, 
                                   metric='precomputed', 
                                   algorithm='brute')
        
        knn.fit(D[np.ix_(train_idx, train_idx)], y_train)
        
        # 5. Predict
        y_pred = knn.predict(D_train_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    return np.mean(accuracies)

def run_decoding(t_ind, diss_arr_3d, trial_labels):
    
    accuracy = decode_spikeship_knn(diss_arr_3d[t_ind, :, :], trial_labels)
    
    return accuracy

def clean_spikeship_nans(diss_arr, max_penalty_factor=1.1):
    """
    Imputes NaNs in a SpikeShip matrix based on biological logic.
    
    Logic:
    - If Row i is ALL NaNs -> Trial i was silent.
    - Silent vs Silent -> Distance 0 (Identity).
    - Silent vs Active -> Distance = Max_Observed * Penalty (High Dissimilarity).
    """
    # Create a copy to avoid modifying the original
    D = diss_arr.copy()
    n_trials = D.shape[0]
    
    # 1. Identify "Silent" trials
    # If a trial has 0 spikes, it cannot have a flow to ANY other trial, 
    # so its entire row (excluding diagonal) is likely NaN.
    # We check if >90% of the row is NaN to be safe.
    is_silent = np.isnan(D).sum(axis=1) > (n_trials * 0.9)
    
    # 2. Get the maximum valid distance in the matrix to use as a penalty
    # We ignore NaNs for this calculation
    valid_max = np.nanmax(D)
    penalty_value = valid_max * max_penalty_factor
    
    # 3. Iterate and Impute
    # (Vectorization is possible but this loop is clearer for the logic)
    for i in range(n_trials):
        for j in range(n_trials):
            if np.isnan(D[i, j]):
                if i == j:
                    # Diagonal is always 0
                    D[i, j] = 0.0
                elif is_silent[i] and is_silent[j]:
                    # Both silent -> They look the same
                    D[i, j] = 0.0
                else:
                    # One silent, one active -> Maximally different
                    D[i, j] = penalty_value
                    
    return D
    
# %% Loop over recordings

decode_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\n{subject} {date} {probe} ({i} of {rec.shape[0]})')

    # Load in data
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)    
    if trials.shape[0] < MIN_TRIALS:
        continue
    
    
    # %% Loop over regions
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        print(f'Starting {region}')
        
        # Get region neurons
        region_neurons = clusters['cluster_id'][clusters['region'] == region]
        region_spikes = spikes['times'][np.isin(spikes['clusters'], region_neurons)]
        region_clusters = spikes['clusters'][np.isin(spikes['clusters'], region_neurons)]
        
        if np.unique(region_clusters).shape[0] < MIN_NEURONS:
            continue
        
        # Loop over objects
        for obj in [1, 2]:
            
            # Run SpikeShip in parallel over timebins
            trial_ids = all_obj_df.loc[all_obj_df['object'] == obj, 'goal'].values
            trial_times = all_obj_df.loc[all_obj_df['object'] == obj, 'times'].values
            results = Parallel(n_jobs=N_CPUS)(
                delayed(run_spikeship)(bin_center, region_spikes, region_clusters, trial_times)
                for bin_center in t_centers)
            
            # Clean up NaNs (trials with too few spikes)
            clean_results = [clean_spikeship_nans(i) for i in results]
            diss_arr_3d = np.array(clean_results)
            
            # Do decoding
            results = Parallel(n_jobs=N_CPUS)(
                delayed(run_decoding)(t_ind, diss_arr_3d, trial_ids) for t_ind in range(diss_arr_3d.shape[0]))
            accuracy = np.array(results)
            
            # Add to dataframe
            decode_df = pd.concat((decode_df, pd.DataFrame(data={
                'accuracy': accuracy, 'time': t_centers, 'region': region, 'object': obj,
                'subject': subject, 'date': date, 'probe': probe
                })))
    
    # Save output
    decode_df.to_csv(path_dict['save_path'] / 'decode_spikeship.csv', index=False)
       
        