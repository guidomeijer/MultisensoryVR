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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from brainbox.population.decode import classify
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, load_subjects, get_spike_counts_in_bins

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
D_BEFORE = 10  # s
D_AFTER = 160
BIN_SIZE = 5
STEP_SIZE = 1
N_NEURONS = 30
N_NEURON_PICKS = 100
MIN_SPEED = 50  # mm/s
ONLY_GOOD_NEURONS = True

# Create time array
d_centers = np.arange(-D_BEFORE + (BIN_SIZE/2), D_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
#clf = RandomForestClassifier(random_state=42)
#clf = GaussianNB()
clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE, histology=True,
                                              only_good=ONLY_GOOD_NEURONS)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))
wheel_speed = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.wheelSpeed.npy'))
wheel_dist = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.wheelDistance.npy'))
wheel_times = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.times.npy'))

# Set a speed threshold, find for each spike its corresponding speed
indices = np.searchsorted(wheel_times, spikes['times'], side='right') - 1
indices = np.clip(indices, 0, wheel_dist.shape[0] - 1)
spike_speed = wheel_speed[indices]
spikes_dist = spikes['distances'][spike_speed >= MIN_SPEED]
clusters_dist = spikes['clusters'][spike_speed >= MIN_SPEED]

# Convert from mm to cm
spikes_dist = spikes_dist / 10
trials['enterEnvPos'] = trials['enterEnvPos'] / 10


# %% Functions

def classify_subselection(spike_counts, n_neurons, trial_labels, clf, cv):
    
    # Subselect neurons
    these_neurons = np.random.choice(np.arange(spike_counts.shape[1]), N_NEURONS, replace=False)

    # Decode goal vs distractor
    accuracy, _, _ = classify(spike_counts[:, these_neurons], trial_labels, clf, cross_validation=cv)
    
    return accuracy
    

# %% Loop over distance bins
decode_df, shuffles_df = pd.DataFrame(), pd.DataFrame()
for i, bin_center in enumerate(d_centers):
    if np.mod(i, 10) == 0:
        print(f'Distance bin {np.round(bin_center, 2)} cm ({i} of {len(d_centers)})')
    
    # Get spike counts per trial for all neurons during this time bin
    these_intervals = np.vstack((trials['enterEnvPos'] + (bin_center - (BIN_SIZE/2)),
                                 trials['enterEnvPos'] + (bin_center + (BIN_SIZE/2)))).T
    spike_counts, neuron_ids = get_spike_counts_in_bins(spikes_dist, clusters_dist,
                                                        these_intervals)
    spike_counts = spike_counts.T  # transpose array into [trials x neurons]
    
    # Loop over regions
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        if np.sum(clusters['region'] == region) < N_NEURONS:
            continue        
        
        # Select neurons from this region and trials of this object
        region_counts = spike_counts[:, clusters['region'] == region]
        
        # Do decoding with random subselection of neurons, use parallel processing
        results = Parallel(n_jobs=-1)(
            delayed(classify_subselection)(region_counts, N_NEURONS, trials['soundId'].values, clf, kfold_cv)
            for i in range(N_NEURON_PICKS))
        accuracy = np.array([result for result in results])
        
        # Add to dataframe
        decode_df = pd.concat((decode_df, pd.DataFrame(data={
            'time': bin_center, 'accuracy': accuracy, 'region': region})))
        
    # Save to disk
    decode_df.to_csv(join(path_dict['save_path'], 'decode_context_environment.csv'), index=False)
            
            
            
            
        
    