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
D_BEFORE = 0  # cm
D_AFTER = 150
LDA_PLACE = 20
BIN_SIZE = 5
STEP_SIZE = 1
MIN_NEURONS = 10
MIN_SPEED = 25  # mm/s

# Create distance array
d_centers = np.arange(-D_BEFORE + (BIN_SIZE/2), D_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
subjects = load_subjects()
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
lda, lda_control = dict(), dict()
lda_dist_df = pd.DataFrame()

for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'{subject} | {date} | {probe} | {i} of {len(rec)}')
    
    # Load in data for this session
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', subject, date, 'trials.csv'))
    spikes, clusters, channels = load_neural_data(session_path, probe)
    wheel_speed = np.load(join(path_dict['local_data_path'], 'Subjects', subject, date, 'continuous.wheelSpeed.npy'))
    wheel_dist = np.load(join(path_dict['local_data_path'], 'Subjects', subject, date, 'continuous.wheelDistance.npy'))
    wheel_times = np.load(join(path_dict['local_data_path'], 'Subjects', subject, date, 'continuous.times.npy'))
        
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
    spike_counts, neuron_ids = get_spike_counts_in_bins(spikes_dist, clusters_dist,
                                                        lda_intervals)
    spike_counts = spike_counts.T  # transpose array into [trials x neurons]
    
       
    # Fit LDA projection per region
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        if np.sum(clusters['region'] == region) < MIN_NEURONS:
            continue
        
        # In environment
        region_counts = spike_counts[:, np.isin(neuron_ids, clusters['cluster_id'][clusters['region'] == region])]
        lda[region] = LinearDiscriminantAnalysis()
        lda[region].fit(region_counts, trials['soundId'])
        
    
    # Loop over all distance bins
    for i, bin_center in enumerate(d_centers):
        
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
            
            # Project neural activity to LDA projection
            region_counts = spike_counts[:, np.isin(neuron_ids, clusters['cluster_id'][clusters['region'] == region])]
            lda_proj = lda[region].transform(region_counts)
            lda_dist = np.abs(np.mean(lda_proj[trials['soundId'] == 1]) - 
                              np.mean(lda_proj[trials['soundId'] == 2]))
            
            # Add to dataframe
            lda_dist_df = pd.concat((lda_dist_df, pd.DataFrame(index=[lda_dist_df.shape[0]], data={
                'distance': bin_center, 'lda_distance': lda_dist, 'region': region})))
            
        # Save to disk
        lda_dist_df.to_csv(join(path_dict['save_path'], 'lda_distance_context.csv'), index=False)
            
            
            
            
        
    