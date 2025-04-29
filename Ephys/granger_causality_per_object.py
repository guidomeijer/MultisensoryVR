# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from brainbox.population.decode import get_spike_counts_in_bins, classify
from joblib import Parallel, delayed
from msvr_functions import (paths, load_neural_data, load_subjects, load_objects,
                            calculate_peths)

# Settings
T_BEFORE = 2  # s
T_AFTER = 2
BIN_SIZE = 0.05
SMOOTHING = 0.2
MIN_NEURONS = 10
MIN_TRIALS = 30
N_CORES = -1

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))


# Function for parallelization
def lda_distance(t, X, y):
        
    X_t = X[:, :, t]  # shape: (n_trials, n_neurons)
    lda_distances = np.empty(X.shape[0])  # shape: (n_trials)
    
    for i in range(X.shape[0]):  # Loop over trials
        # Leave one trial out
        X_train = np.delete(X_t, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X_t[i].reshape(1, -1)
        
        # Fit LDA
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X_train, y_train)
        
        # Project held-out trial
        proj_test = lda.transform(X_test)[0, 0]
        
        # Get class means from training set
        proj_0 = lda.transform(X_train[y_train == 0])
        proj_1 = lda.transform(X_train[y_train == 1])
        pooled_std = np.sqrt(0.5 * (proj_0.var() + proj_1.var()))
        center = (proj_0.mean() + proj_1.mean()) / 2
        
        # Get class means from training set
        center = (lda.transform(X_train[y_train == 0]).mean()
                  + lda.transform(X_train[y_train == 1]).mean()) / 2
        
        # Compute absolute distance from the decision boundary divided by the pooled variance
        lda_distances[i] = np.abs((proj_test - center) / pooled_std)
                
    return lda_distances


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
        if region_neurons.shape[0] < MIN_NEURONS:
            continue
        
        # Sound A versus sound B at object 1
        _, obj1_soundA = calculate_peths(
            spikes['times'], spikes['clusters'], region_neurons, 
            all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['sound'] == 1), 'times'].values,
            T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
            )
        _, obj1_soundB = calculate_peths(
            spikes['times'], spikes['clusters'], region_neurons, 
            all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['sound'] == 2), 'times'].values,
            T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
            )
        
        # Prepare data for LDA
        X = np.concatenate([obj1_soundA, obj1_soundB], axis=0)  # shape: (n_trials_total, n_neurons, n_timebins)
        y = np.concatenate([np.zeros(obj1_soundA.shape[0]), np.ones(obj1_soundB.shape[0])])  # 1 = sound A, 2 = sound B
        
        # Fit LDA projection on all data except trial i, project trial i on fitted axis and get distance
        # Compute LDA distance with parallel processing over timebins
        results = Parallel(n_jobs=N_CORES)(delayed(lda_distance)(t, X, y) for t in range(X.shape[2]))
        lda_distances = np.vstack(results).T  # shape: (trials x timebins)
        
      
    