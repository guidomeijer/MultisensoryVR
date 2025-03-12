# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, load_subjects, get_spike_counts_in_bins

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
D_BEFORE = 0  # s
D_AFTER = 40
LDA_PLACE = 130
LDA_CONTROL = 0
BIN_SIZE = 5
STEP_SIZE = 1
MIN_NEURONS = 10
MIN_SPEED = 50  # mm/s
ONLY_GOOD_NEURONS = True

# Create distance array
d_centers = np.arange(-D_BEFORE + (BIN_SIZE/2), D_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
lda, lda_control = dict(), dict()

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


# %% 

# Get neural activity of a bin in the environment
lda_intervals = np.vstack((trials['enterEnvPos'] + (LDA_PLACE - (BIN_SIZE/2)),
                           trials['enterEnvPos'] + (LDA_PLACE + (BIN_SIZE/2)))).T
spike_lda, neuron_ids = get_spike_counts_in_bins(spikes_dist, clusters_dist,
                                                    lda_intervals)
spike_lda = spike_lda.T  # transpose array into [trials x neurons]

# Get neural activity of a bin in the tunnel
control_intervals = np.vstack((trials['enterEnvPos'] + (LDA_CONTROL - (BIN_SIZE/2)),
                               trials['enterEnvPos'] + (LDA_CONTROL + (BIN_SIZE/2)))).T
spike_control, neuron_ids = get_spike_counts_in_bins(spikes_dist, clusters_dist,
                                                    control_intervals)
spike_control = spike_control.T  # transpose array into [trials x neurons]

# Loop over all distance bins
lda_dist_df, shuffles_df = pd.DataFrame(), pd.DataFrame()
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
        if np.sum(clusters['region'] == region) < MIN_NEURONS:
            continue        
        
        # Fit LDA projection
        lda = LinearDiscriminantAnalysis()
        lda.fit(spike_lda[:, clusters['region'] == region], trials['soundId'].values)
        lda_proj = lda.transform(spike_counts[:, clusters['region'] == region])
        lda_dist = np.abs(np.mean(lda_proj[trials['soundId'] == 1]) - 
                          np.mean(lda_proj[trials['soundId'] == 2]))
        
        # For control projection
        lda_control = LinearDiscriminantAnalysis()
        lda_control.fit(spike_control[:, clusters['region'] == region], trials['soundId'].values)
        lda_proj_control = lda_control.transform(spike_counts[:, clusters['region'] == region])
        lda_dist_control = np.abs(np.mean(lda_proj_control[trials['soundId'] == 1]) - 
                                  np.mean(lda_proj_control[trials['soundId'] == 2]))
        
        # Add to dataframe
        lda_dist_df = pd.concat((lda_dist_df, pd.DataFrame(data={
            'distance': bin_center, 'lda_distance': [lda_dist, lda_dist_control], 'region': region,
            'control': [0, 1]})))
        
    # Save to disk
    lda_dist_df.to_csv(join(path_dict['save_path'], 'lda_distance_context_sound_onset.csv'), index=False)
            
            
            
            
        
    