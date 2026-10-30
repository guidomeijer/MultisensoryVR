# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2026

Bin neural activity per spatial bin and normalize by the time spent in each bin
(i.e. firing rate in spikes/s), without any GLM fitting.

By Guido Meijer
"""

import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, bin_signal

# Settings
BIN_SIZE = 50  # mm
STEP_SIZE = 10
MIN_TRIALS = 0
MIN_SPEED = 20  # mm/s
N_CORES = -8
OVERWRITE = True

# Create position bins
rel_bin_centers = np.arange((BIN_SIZE/2), 1500 - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

def process_recording(i, subject, date, probe, n_recs, path_dict, rel_bin_centers):
    print(f'\n{subject} {date} {probe} ({i} of {n_recs})')

    # Load in data
    session_path = path_dict['local_data_path'] / 'subjects' / f'{subject}' / f'{date}'
    trials = pd.read_csv(session_path / 'trials.csv')
    if trials.shape[0] < MIN_TRIALS:
        print('Too few trials, skipping session')
        return None

    spikes, clusters, channels = load_neural_data(session_path, probe)
    position = np.load(session_path / 'continuous.wheelDistance.npy')
    running_speed = np.load(session_path / 'continuous.wheelSpeed.npy')
    timestamps = np.load(session_path / 'continuous.times.npy')

    # Filter spikes by speed
    valid_spike_mask = spikes['speeds'] > MIN_SPEED

    # Time per sample, only counting samples above the speed threshold so that
    # occupancy matches the speed-filtered spikes
    dt = np.concatenate((np.diff(timestamps), [np.mean(np.diff(timestamps))]))
    dt[running_speed <= MIN_SPEED] = 0

    # Create a dictionary of valid spike POSITIONS, keyed by neuron_id
    spikes_by_neuron = {}
    for neuron_id in clusters['cluster_id']:
        neuron_mask = np.isin(spikes['clusters'], neuron_id) & valid_spike_mask
        spikes_by_neuron[neuron_id] = spikes['distances'][neuron_mask]

    # Loop over trials
    binned_spikes = {neuron_id: [] for neuron_id in clusters['cluster_id']}
    time_per_bin_list, trial_id_list, rel_pos_list = [], [], []
    for row in trials.itertuples():
        if row.soundId == 0:
            continue

        # Define spatial bins
        abs_bin_centers = rel_bin_centers + row.enterEnvPos
        rel_pos_list.append(rel_bin_centers)
        trial_id_list.append(np.full(abs_bin_centers.shape[0], row.Index))

        # Time occupancy
        time_per_bin_list.append(bin_signal(position, dt, abs_bin_centers, BIN_SIZE,
                                            statistic='sum'))

        # Spike counts per neuron
        for neuron_id, neuron_spike_pos in spikes_by_neuron.items():
            trial_spike_mask = ((neuron_spike_pos >= row.enterEnvPos)
                                & (neuron_spike_pos <= row.exitEnvPos))
            binned_spikes[neuron_id].append(bin_signal(
                position, neuron_spike_pos[trial_spike_mask], abs_bin_centers, BIN_SIZE,
                statistic='count'))

    # Normalize spike counts by time spent in each bin (bins without occupancy become NaN)
    occupancy = np.concatenate(time_per_bin_list)
    spike_counts = np.vstack([np.concatenate(binned_spikes[n])
                              for n in clusters['cluster_id']]).T  # bins x neurons
    with np.errstate(divide='ignore', invalid='ignore'):
        firing_rate = spike_counts / occupancy[:, None]
    firing_rate[occupancy == 0, :] = np.nan

    trial_id = np.concatenate(trial_id_list)

    return {
        'firing_rate': firing_rate,
        'occupancy': occupancy,
        'neuron_id': np.array(clusters['cluster_id']),
        'region': clusters['region'],
        'acronym': clusters['acronym'],
        'position': np.concatenate(rel_pos_list),
        'trial': trial_id,
        'context': np.array([trials.loc[i, 'soundId'] for i in trial_id]),
        'subject': subject,
        'date': date,
        'probe': probe
    }

if __name__ == '__main__':
    # Initialize
    path_dict = paths(sync=False)
    rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
    save_path = path_dict['google_drive_data_path'] / f'firing_rates_position_{MIN_SPEED}mms.pickle'

    if OVERWRITE:
        fr_dict = {
            'firing_rate': [],
            'occupancy': [],
            'neuron_id': [],
            'region': [],
            'acronym': [],
            'position': [],
            'trial': [],
            'context': [],
            'subject': [],
            'date': [],
            'probe': []
            }
    else:
        with open(save_path, 'rb') as handle:
            fr_dict = pickle.load(handle)
        match_cols = ['subject', 'date', 'probe']
        fr_df = pd.DataFrame({c: fr_dict[c] for c in match_cols}).astype(str)
        rec[match_cols] = rec[match_cols].astype(str)
        merged = rec.merge(fr_df, on=match_cols, how='left', indicator=True)
        rec = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

    # Loop over recordings
    print(f'Processing {rec.shape[0]} recordings')
    results = Parallel(n_jobs=N_CORES)(
        delayed(process_recording)(i, subject, date, probe, rec.shape[0], path_dict, rel_bin_centers)
        for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe']))
    )

    # Add results to dictionary
    for result in results:
        if result is not None:
            for key in fr_dict.keys():
                fr_dict[key].append(result[key])

    # Save to disk
    with open(save_path, 'wb') as handle:
        pickle.dump(fr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
