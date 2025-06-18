# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from brainbox.population.decode import get_spike_counts_in_bins, classify
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, load_subjects, load_objects

# Settings
T_BEFORE = 2  # s
T_AFTER = 2
BIN_SIZE = 0.3
STEP_SIZE = 0.025
MIN_NEURONS = 10
MIN_TRIALS = 30
N_CORES = 6

# Create time array
t_centers = np.arange(-T_BEFORE + (BIN_SIZE/2), T_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

clf = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=4, min_samples_leaf=5)

# Function for parallelization
def decode_context(bin_center, spikes, region_neurons, trials):
    
    # Get spike counts per trial for all neurons during this time bin
    these_intervals = np.vstack((trials['soundOnsetTime'] + (bin_center - (BIN_SIZE/2)),
                                 trials['soundOnsetTime'] + (bin_center + (BIN_SIZE/2)))).T
    spike_counts, neuron_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'],
                                                        these_intervals)
    spike_counts = spike_counts.T  # transpose array into [trials x neurons]
        
    # Select neurons from this region and trials of this object
    use_counts = spike_counts[:, np.isin(neuron_ids, region_neurons)]
        
    # Do decoding 
    accuracy, _, _ = classify(use_counts, trials['soundId'], clf, cross_validation=kfold_cv)
    
    return accuracy


decode_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\n{subject} {date} {probe} ({i} of {rec.shape[0]})')

    # Load in data
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    #spikes, clusters, channels = load_neural_data(session_path, probe)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    print(f'{np.round(np.median(trials["soundOnsetTime"] - trials["enterEnvTime"]), 2)}')
    
    
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
        
        # Do decoding with with parallel processing
        results = Parallel(n_jobs=N_CORES)(
            delayed(decode_context)(bin_center, spikes, region_neurons, trials) for bin_center in t_centers)
        
        # Add to dataframe
        decode_df = pd.concat((decode_df, pd.DataFrame(data={
            'time': t_centers, 'accuracy': [i for i in results], 
            'region': region, 'subject': subject, 'date': date, 'probe': probe})))
         
# Save to disk
decode_df.to_csv(join(path_dict['save_path'], 'decode_context_onset.csv'), index=False)

