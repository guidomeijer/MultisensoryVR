# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:28:35 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from pathlib import Path
from msvr_functions import paths, load_multiple_probes, get_spike_counts_in_bins, bin_signal

# Settings
BIN_SIZE = 0.05   # s
DATA_PATH = Path(r'V:\imaging1\guido')
SAVE_PATH = Path(r'D:\MultisensoryVR\Nikki')

# Initialize
rec = pd.read_csv(DATA_PATH / 'recordings.csv', dtype={'subject': str, 'date': str})
rec = rec.drop_duplicates(subset=['subject', 'date'], keep='first')

# Loop over recordings
for i, row in rec.iterrows():
    print(f'Processing {row["subject"]} {row["date"]}')
    
    # Load in session data
    ses_path = path_dict['local_data_path'] / 'Subjects' / row["subject"] / row["date"]
    spikes, clusters, _ = load_multiple_probes(ses_path)
    trials = pd.read_csv(ses_path / 'trials.csv')
    position = np.load(ses_path / 'continuous.wheelDistance.npy')
    position = position / 10  # convert from mm to cm
    speed = np.load(ses_path / 'continuous.wheelSpeed.npy')
    sniffing = np.load(ses_path / 'continuous.breathing.npy')
    lick_times = np.load(ses_path / 'lick.times.npy')
    reward_times = np.load(ses_path / 'reward.times.npy')
    timestamps = np.load(ses_path / 'continuous.times.npy')
    pupil_times = np.load(ses_path / 'camera.times.npy')
    if (ses_path / 'pupil.csv').exists():
        pupil_size = pd.read_csv(ses_path / 'pupil.csv')
        if pupil_times.shape[0] > pupil_size.shape[0]:
            pupil_times = pupil_times[:pupil_size.shape[0]]
        else:
            pupil_size = pupil_size.iloc[:pupil_times.shape[0]]
    else:
        pupil_size = np.full(pupil_times.shape[0], np.nan)
    
    # Initialize empty arrays 
    max_trial_dur = np.max(trials['exitEnvTime'] - trials['enterEnvTime'])
    n_bins = int(max_trial_dur / BIN_SIZE) + 1
    n_trials = trials.shape[0]
    bin_centers = np.zeros((n_trials, n_bins))
    controls = np.zeros((n_trials, n_bins, 4))
    behavior = np.zeros((n_trials, n_bins, 4))
    trial_mask = np.zeros((n_trials, n_bins)).astype(int)

    # Initialize neuron data
    n_neurons = np.sum([clusters[i]['cluster_id'].shape[0] for i in clusters.keys()])
    neuron_probes = np.concatenate([np.array([i] * clusters[i]['cluster_id'].shape[0]) for i in clusters.keys()])
    region_id = np.concatenate([clusters[i]['region'] for i in clusters.keys()])
    neuron_id = np.concatenate([clusters[i]['cluster_id'] for i in clusters.keys()])
    spike_counts = np.zeros((n_trials, n_bins, n_neurons)).astype(int)

    # Loop over trials
    for t, trial in trials.iterrows():

        # Create bin edges for this trial
        trial_bin_edges = np.arange(trial['enterEnvTime'], trial['exitEnvTime'] + BIN_SIZE, BIN_SIZE)
        trial_intervals = np.column_stack((trial_bin_edges[:-1], trial_bin_edges[1:]))
        trial_bin_centers = trial_bin_edges[:-1] + BIN_SIZE / 2
        trial_n_bins = trial_bin_centers.shape[0]

        # Get spike counts for this trial, loop over both probes to get all neural data for this session
        for probe in clusters.keys():
            probe_spike_counts, _ = get_spike_counts_in_bins(
                spikes[probe]['times'], spikes[probe]['clusters'], trial_intervals)
            spike_counts[t, :trial_n_bins, neuron_probes == probe] = probe_spike_counts.astype(int)
            
        # Get other variables for this trial
        trial_position = bin_signal(timestamps, position - trial['enterEnvPos'], trial_bin_centers, BIN_SIZE)
        trial_speed = bin_signal(timestamps, speed, trial_bin_centers, BIN_SIZE)
        trial_sniffing = bin_signal(timestamps, sniffing, trial_bin_centers, BIN_SIZE)
        trial_licks = bin_signal([], lick_times, trial_bin_centers, BIN_SIZE, statistic='count')
        trial_rewards = bin_signal([], reward_times, trial_bin_centers, BIN_SIZE, statistic='count')
        if (ses_path / 'pupil.csv').exists():
            trial_pupil = bin_signal(pupil_times, pupil_size['width_smooth'], trial_bin_centers, BIN_SIZE)
        else:
            trial_pupil = np.full(trial_n_bins, np.nan)
        
        # Get object occupancy
        trial_objects = np.zeros(trial_n_bins)
        trial_objects[(trial_bin_centers >= trial['enterObj1Time']) & (trial_bin_centers <= trial['exitObj1Time'])] = 1
        trial_objects[(trial_bin_centers >= trial['enterObj2Time']) & (trial_bin_centers <= trial['exitObj2Time'])] = 2
        trial_objects[(trial_bin_centers >= trial['enterObj3Time']) & (trial_bin_centers <= trial['exitObj3Time'])] = 3

        # Add to arrays
        bin_centers[t, :trial_n_bins] = trial_bin_centers - trial_bin_centers[0] + (BIN_SIZE / 2)
        trial_mask[t, :trial_n_bins] = 1
        controls[t, :trial_n_bins, 0] = trial_position
        controls[t, :trial_n_bins, 1] = trial_rewards.astype(int)
        controls[t, :trial_n_bins, 2] = trial_objects.astype(int)
        controls[t, :trial_n_bins, 3] = trial['soundId'].astype(int)
        behavior[t, :trial_n_bins, 0] = trial_speed
        behavior[t, :trial_n_bins, 1] = trial_licks.astype(int)
        behavior[t, :trial_n_bins, 2] = trial_pupil
        behavior[t, :trial_n_bins, 3] = trial_sniffing
    
    # Save data to file
    Path.mkdir(SAVE_PATH / row['subject'], exist_ok=True)
    Path.mkdir(SAVE_PATH / row['subject'] / row['date'], exist_ok=True)
    np.save(SAVE_PATH / row['subject'] / row['date'] / 'times.npy', bin_centers)
    np.save(SAVE_PATH / row['subject'] / row['date'] / 'trial_mask.npy', trial_mask)
    np.save(SAVE_PATH / row['subject'] / row['date'] / 'controls.npy', controls)
    np.save(SAVE_PATH / row['subject'] / row['date'] / 'behavior.npy', behavior)
    np.save(SAVE_PATH / row['subject'] / row['date'] / 'spikes.npy', spike_counts)
    np.save(SAVE_PATH / row['subject'] / row['date'] / 'region_id.npy', region_id)
