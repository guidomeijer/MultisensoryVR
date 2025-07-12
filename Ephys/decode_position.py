# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, load_subjects, load_objects

def calculate_occupancy(animal_position, timestamps, track_length, bin_size, speed, speed_threshold):
    """
    Calculates the time spent in each position bin (occupancy).
    """
    position_bins = np.arange(0, track_length + bin_size, bin_size)
    bin_centers = position_bins[:-1] + bin_size / 2
    moving_indices = np.where(speed >= speed_threshold)[0]
    moving_positions = animal_position[moving_indices] % track_length
    dt = np.mean(np.diff(timestamps))
    occupancy_counts, _ = np.histogram(moving_positions, bins=position_bins)
    occupancy = occupancy_counts * dt
    occupancy = occupancy + 1e-9
    return occupancy, bin_centers

def calculate_firing_rate_maps(spikes, trials, animal_position, timestamps, track_length, bin_size, speed, speed_threshold, use_neurons=None):
    """
    Calculates firing rate maps using data only from within defined trial periods.
    """
    if use_neurons is None:
        neurons_to_process = np.unique(spikes['clusters'])
    else:
        existing_neurons = np.unique(spikes['clusters'])
        neurons_to_process = sorted([n for n in use_neurons if n in existing_neurons])

    position_bins = np.arange(0, track_length + bin_size, bin_size)
    num_bins = len(position_bins) - 1
    dt = np.mean(np.diff(timestamps))

    total_occupancy_time = np.zeros(num_bins)
    total_spike_counts = np.zeros((len(neurons_to_process), num_bins))

    # Aggregate data across all trials
    for i in range(len(trials['enterEnvTime'])):
        # Get mask for data within this trial's environment period
        trial_mask = (timestamps >= trials['enterEnvTime'][i]) & (timestamps < trials['exitEnvTime'][i]) & (speed >= speed_threshold)
        
        # Normalize position for this trial
        pos_in_trial = animal_position[trial_mask] - trials['enterEnvPos'][i]
        
        # Calculate occupancy for this trial
        occupancy_counts, _ = np.histogram(pos_in_trial, bins=position_bins)
        total_occupancy_time += (occupancy_counts * dt)
        
        # Get spikes for this trial
        spike_mask = (spikes['times'] >= trials['enterEnvTime'][i]) & (spikes['times'] < trials['exitEnvTime'][i])
        trial_spikes_clusters = spikes['clusters'][spike_mask]
        trial_spikes_pos = spikes['distances'][spike_mask] - trials['enterEnvPos'][i]

        for n_idx, neuron_id in enumerate(neurons_to_process):
            neuron_spike_pos = trial_spikes_pos[trial_spikes_clusters == neuron_id]
            spike_counts, _ = np.histogram(neuron_spike_pos, bins=position_bins)
            total_spike_counts[n_idx, :] += spike_counts

    # Calculate firing rate: total spikes / total time
    total_occupancy_time += 1e-9 # Avoid division by zero
    firing_rate_maps = total_spike_counts / total_occupancy_time
    
    # Smooth maps
    for i in range(len(neurons_to_process)):
        firing_rate_maps[i, :] = np.convolve(firing_rate_maps[i, :], np.ones(5)/5, mode='same')
        
    bin_centers = position_bins[:-1] + bin_size / 2
    occupancy_prior = total_occupancy_time / np.sum(total_occupancy_time)
    
    return firing_rate_maps, occupancy_prior, bin_centers

def bayesian_decode(spike_counts_in_window, firing_rate_maps, occupancy_prior, time_window_duration):
    """Decodes position using a Bayesian algorithm."""
    log_likelihood = np.dot(spike_counts_in_window, np.log(firing_rate_maps + 1e-9)) - time_window_duration * np.sum(firing_rate_maps, axis=0)
    log_prior = np.log(occupancy_prior + 1e-9)
    log_posterior = log_likelihood + log_prior
    posterior = np.exp(log_posterior - np.max(log_posterior))
    return posterior / np.sum(posterior)

def parallel_process(time_window_idx, start_times, spikes, use_neurons, firing_rate_maps,
                     occupancy_prior):
    start_time = start_times[time_window_idx]
    end_time = start_time + TIME_WINDOW         
    window_spike_indices = np.where((spikes['times'] >= start_time)
                                    & (spikes['times'] < end_time))[0]
    window_spike_clusters = spikes['clusters'][window_spike_indices]


    spike_counts = np.array([np.sum(window_spike_clusters == nid) for nid in use_neurons])
    posterior = bayesian_decode(spike_counts, firing_rate_maps, occupancy_prior, TIME_WINDOW)
    
    # Find which trial this time window belongs to
    time_mid_point = start_time + TIME_WINDOW / 2
    trial_idx = np.where((trials['enterEnvTime'] <= time_mid_point)
                         & (trials['exitEnvTime'] > time_mid_point))[0][0]
    decoded_position = bin_centers[np.argmax(posterior)]
    actual_pos_idx = np.argmin(np.abs(timestamps - time_mid_point))
    actual_position_in_env = animal_position[actual_pos_idx] - trials['enterEnvPos'][trial_idx]
    
    return decoded_position, time_mid_point, actual_position_in_env

# %% MAIN SCRIPT

# Settings
MIN_NEURONS = 5
TRACK_LENGTH = 1500  # mm
BIN_SIZE = 50       # mm
SPEED_THRESHOLD = 10 # mm/s
TIME_WINDOW = 0.250  # 250 ms for decoding
N_CPUS = -1

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# %% Loop over recordings

error_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\n{subject} {date} {probe} ({i} of {rec.shape[0]})')

    # Load in data
    session_path = path_dict['local_data_path'] / 'Subjects' / subject / date
    spikes, clusters, channels = load_neural_data(session_path, probe)
    trials = pd.read_csv(session_path / 'trials.csv')
    all_obj_df = load_objects(subject, date)
    animal_position = np.load(session_path / 'continuous.wheelDistance.npy')
    animal_speed = np.load(session_path / 'continuous.wheelSpeed.npy')
    timestamps = np.load(session_path / 'continuous.times.npy')
    
        
    # %% Loop over regions
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        print(f'\nStarting {region}')
        
        # Get region neurons
        use_neurons = clusters['cluster_id'][clusters['region'] == region]
        if use_neurons.shape[0] < MIN_NEURONS:
            continue
                
        # --- 3. Select Neurons and Build Encoding Model ---
        all_neuron_ids = np.unique(spikes['clusters'])
        print(f"Using {len(use_neurons)} neurons for decoding.")
        
        firing_rate_maps, occupancy_prior, bin_centers = calculate_firing_rate_maps(
            spikes, trials, animal_position, timestamps, TRACK_LENGTH, BIN_SIZE, animal_speed,
            SPEED_THRESHOLD, use_neurons=use_neurons
        )
        
        # --- 4. Decode Position ---
        print("Decoding position in sliding windows...")
        start_times = np.arange(trials['enterEnvTime'].values[0], trials['exitEnvTime'].values[-1],
                                step=TIME_WINDOW)
        time_mid_points = start_times + TIME_WINDOW / 2
        
        # First determine which time windows are during environments
        is_during_any_trial = np.any(
            (trials['enterEnvTime'].values <= time_mid_points[:, None]) & 
            (trials['exitEnvTime'].values > time_mid_points[:, None]), 
            axis=1
            )
        speed_during_times = animal_speed[np.clip(
            np.searchsorted(timestamps, time_mid_points, side='right') - 1, 0, animal_speed.shape[0] - 1)]
        valid_start_times = start_times[is_during_any_trial & (speed_during_times > SPEED_THRESHOLD)]
     
        # Do decoding with with parallel processing
        results = Parallel(n_jobs=N_CPUS)(
            delayed(parallel_process)(i, valid_start_times, spikes, use_neurons, firing_rate_maps,
                                      occupancy_prior) for i in range(valid_start_times.shape[0]))
        decoded_positions = np.array([i[0] for i in results])
        decoding_times = np.array([i[1] for i in results])
        actual_positions_in_env = np.array([i[2] for i in results])
        error = np.mean(np.abs(decoded_positions - actual_positions_in_env))
        print(f"Mean Absolute Decoding Error: {error:.2f} mm")
        
        # Add to dataframe
        error_df = pd.concat((error_df, pd.DataFrame(index=[error_df.shape[0]], data={
            'error_mm': error, 'region': region, 'subject': subject, 'date': date,
            'n_neurons': use_neurons.shape[0], 'n_trials': trials.shape[0]})))
        
        # Save results of this decoding run
        decode_df = pd.DataFrame(data={
            'decoded_positions': decoded_positions, 'decoding_times': decoding_times,
            'actual_positions_in_env': actual_positions_in_env})
        decode_df.to_csv(path_dict['local_data_path'] / 'PositionDecoding' / f'{region}_{subject}_{date}.csv',
                         index=False)
    
    # Save results
    error_df.to_csv(path_dict['save_path'] / 'position_decoding_error.csv', index=False)
    
       