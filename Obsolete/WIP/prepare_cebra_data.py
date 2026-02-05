# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:39:26 2023

By Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import gzip, pickle
from msvr_functions import paths, load_neural_data, bin_signal, get_spike_counts_in_bins

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
MIN_SPEED = 50  # mm/s
BIN_SIZE = 0.025  # s

# Get paths
path_dict = paths()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))
wheel_speed = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.wheelSpeed.npy'))
wheel_dist = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.wheelDistance.npy'))
wheel_times = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.times.npy'))

"""
# Set a speed threshold, find for each spike its corresponding speed
indices = np.searchsorted(wheel_times, spikes['times'], side='right') - 1
indices = np.clip(indices, 0, wheel_dist.shape[0] - 1)
spike_speed = wheel_speed[indices]
spikes_dist = spikes['distances'][spike_speed >= MIN_SPEED]
clusters_dist = spikes['clusters'][spike_speed >= MIN_SPEED]

# Convert from mm to cm
spikes_dist = spikes_dist / 10
trials['enterEnvPos'] = trials['enterEnvPos'] / 10
"""

# Reformat distance into relative distance in the environment
rel_dist_list, count_list, time_list, speed_list = [], [], [], []
for i in trials.index.values[1:]:
    
    # Get relative distance in this trial
    trial_ind = ((wheel_times >= trials.loc[i, 'enterEnvTime'])
                 & (wheel_times <= trials.loc[i, 'exitEnvTime']))
    rel_dist = wheel_dist[trial_ind] - trials.loc[i, 'enterEnvPos']
    this_time = wheel_times[trial_ind]   
    this_speed = wheel_speed[trial_ind]
    
    # Bin distance and speed
    bin_edges = np.arange(this_time[0], this_time[-1], BIN_SIZE)
    binned_dist = bin_signal(this_time, rel_dist, bin_edges)
    binned_speed = bin_signal(this_time, this_speed, bin_edges)
    
    # Bin spike counts
    intervals = np.vstack((bin_edges[:-1], bin_edges[1:])).T
    these_counts, neuron_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'], intervals)
    
    # Get time axis as bin centers
    bin_centers = (intervals[:, 0] + intervals[:, 1]) / 2
    
    # Add to lists
    rel_dist_list.append(binned_dist)
    speed_list.append(binned_speed)
    count_list.append(these_counts)
    time_list.append(bin_centers)
    
# Convert lists to np arrays
rel_distance = np.concatenate(rel_dist_list)
speed = np.concatenate(speed_list)
spike_counts = np.hstack(count_list).T  # time x neurons
timestamps = np.concatenate(time_list)

# Apply speed threshold
rel_distance = rel_distance[speed >= MIN_SPEED]
spike_counts = spike_counts[speed >= MIN_SPEED, :]
timestamps = timestamps[speed >= MIN_SPEED]
    
# Add data to dict
data_dict = {'spike_counts': spike_counts,
             'relative_distance': rel_distance,
             'timestamps': timestamps,
             'neuron_ids': neuron_ids,
             'acronyms': clusters['acronym']}

# Save prepared data to disk
with gzip.open(join(path_dict['local_data_path'], 'CEBRA',
                    f'{SUBJECT}_{DATE}_{PROBE}.pickle'), 'wb') as handle:
    pickle.dump(data_dict, handle)

