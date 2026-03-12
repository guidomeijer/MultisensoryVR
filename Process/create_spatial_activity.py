# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 12/03/2026
"""

# %%
import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, bin_signal

# Settings
BIN_SIZE = 50  # mm
STEP_SIZE = 10
MIN_TRIALS = 0
MIN_SPEED = 0  # mm/s
N_CPUS = 18

# Create position bins
rel_bin_centers = np.arange((BIN_SIZE / 2), 1500 - ((BIN_SIZE / 2) - STEP_SIZE), STEP_SIZE)

def process_session(subject, date, probe, path_dict, rel_bin_centers, BIN_SIZE, MIN_TRIALS, MIN_SPEED):
    print(f'Processing {subject} {date} {probe}')
    try:
        # Load in data
        session_path = path_dict['local_data_path'] / 'subjects' / f'{subject}' / f'{date}'
        trials = pd.read_csv(session_path / 'trials.csv')
        if trials.shape[0] < MIN_TRIALS:
            print('Too few trials, skipping session')
            return None

        spikes, clusters, channels = load_neural_data(session_path, probe)
        position = np.load(session_path / 'continuous.wheelDistance.npy')
        timestamps = np.load(session_path / 'continuous.times.npy')

        # Filter spikes by speed
        valid_spike_mask = spikes['speeds'] > MIN_SPEED

        # Create a dictionary of valid spike POSITIONS, keyed by neuron_id
        spikes_by_neuron = {}
        for neuron_id in clusters['cluster_id']:
            # Get spikes for this neuron that also pass the speed threshold
            neuron_mask = np.isin(spikes['clusters'], neuron_id) & valid_spike_mask
            spikes_by_neuron[neuron_id] = spikes['distances'][neuron_mask]

        time_per_bin_list, trial_id_list, binned_spikes, rel_pos_list = [], [], [], []
        for row in trials.itertuples():

            # Define spatial bins
            abs_bin_centers = rel_bin_centers + row.enterEnvPos
            rel_pos_list.append(rel_bin_centers)

            # Get trial id
            trial_id_list.append(np.full(abs_bin_centers.shape[0], row.Index))

            # Time occupancy
            dt = np.concatenate((np.diff(timestamps), [np.mean(np.diff(timestamps))]))
            this_occupancy = bin_signal(position, dt, abs_bin_centers, BIN_SIZE,
                                        statistic='sum')
            time_per_bin_list.append(this_occupancy)

            # --- Get binned spiking activity per neuron ---
            these_binned_spikes = np.full((abs_bin_centers.shape[0], len(spikes_by_neuron)), np.nan)
            for n, (neuron_id, neuron_spike_pos) in enumerate(spikes_by_neuron.items()):
                # Filter spikes for this trial (by position range)
                # They are already filtered by speed globally

                # We need spikes that are within [start_pos, end_pos]
                # Note: neuron_spike_pos are absolute positions
                trial_spike_mask = ((neuron_spike_pos >= row.enterEnvPos)
                                    & (neuron_spike_pos <= row.exitEnvPos))
                these_spike_pos = neuron_spike_pos[trial_spike_mask]

                # Bin spikes
                these_binned_spikes[:, n] = bin_signal(position, these_spike_pos, abs_bin_centers, BIN_SIZE,
                                                       statistic='count')

            # Add to list
            binned_spikes.append(these_binned_spikes)

        return {'binned_spikes': np.vstack(binned_spikes),
                'position': np.concatenate(rel_pos_list),
                'time_per_bin': np.concatenate(time_per_bin_list),
                'trial_id': np.concatenate(trial_id_list),
                'context': np.array([trials.loc[i, 'soundId'] for i in np.concatenate(trial_id_list)]),
                'neuron_id': clusters['cluster_id'],
                'region': clusters['region'],
                'acronym': clusters['acronym'],
                'subject': subject,
                'date': date,
                'probe': probe}
    except Exception as e:
        print(f"Error in {subject} {date} {probe}: {e}")
        return None

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec['subject'] = rec['subject'].astype(str)
rec['date'] = rec['date'].astype(str)
spatial_dict = dict({'binned_spikes': [], 'position': [], 'time_per_bin': [], 'trial_id': [], 'context':[],
                     'neuron_id': [], 'region': [], 'acronym': [], 'subject': [], 'date': [], 'probe': []})

# Parallel execution
results = Parallel(n_jobs=N_CPUS)(delayed(process_session)(
    subject, date, probe, path_dict, rel_bin_centers, BIN_SIZE, MIN_TRIALS, MIN_SPEED)
    for subject, date, probe in zip(rec['subject'], rec['date'], rec['probe']))

# Aggregate results
for res in results:
    if res is None:
        continue
    for k, v in res.items():
        spatial_dict[k].append(v)

# Save to disk
with open(path_dict['google_drive_data_path'] / f'spatial_spikes_{MIN_SPEED}mms.pickle', 'wb') as handle:
    pickle.dump(spatial_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
