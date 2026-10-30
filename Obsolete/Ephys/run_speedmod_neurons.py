# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:45:16 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from joblib import Parallel, delayed
from msvr_functions import (paths, load_neural_data, load_subjects, bin_signal,
                            get_spike_counts_in_bins, circ_shift)

# Settings
OVERWRITE = True
BIN_SIZE = 0.5  # s

# Initialize
path_dict = paths()
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# Load in previous data
if OVERWRITE:
    speed_df = pd.DataFrame()
else:
    speed_df = pd.read_csv(join(path_dict['save_path'], 'speed_neurons.csv'))
    speed_df[['subject', 'date', 'probe']] = speed_df[['subject', 'date', 'probe']].astype(str)
    merged = rec.merge(speed_df, on=['subject', 'date', 'probe'], how='left', indicator=True)
    rec = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    
# %% Function for parallelization

def run_correlation(n, binned_speed, spike_counts):
    r, p, r_null = circ_shift(binned_speed, spike_counts[n, :], n_shifts=1000, min_shift_percentage=0.2)
    return r, p

    
# %%
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\nStarting {subject} {date} {probe}..')
    
    # Load in data
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe)
    wheel_speed = np.load(join(session_path, 'continuous.wheelSpeed.npy'))
    timestamps = np.load(join(session_path, 'continuous.times.npy'))
    
    # Get binned speed and neural activity per neuron
    bin_edges = np.arange(timestamps[0], timestamps[-1], BIN_SIZE)
    if bin_edges.shape[0] % 2 != 0:
        bin_edges = bin_edges[:-1]
    binned_speed = bin_signal(timestamps, wheel_speed, bin_edges)
    intervals = np.vstack((bin_edges[:-1], bin_edges[1:])).T
    spike_counts, neuron_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'], intervals)
        
    # Parallellize over neurons and get significance
    results = Parallel(n_jobs=-1)(
        delayed(run_correlation)(n, binned_speed, spike_counts)
        for n in range(spike_counts.shape[0]))
    r_value = np.array([result[0] for result in results])
    p_value = np.array([result[1] for result in results])
    
    # Add to dataframe
    speed_df = pd.concat((speed_df, pd.DataFrame(data={
        'subject': subject, 'date': date, 'probe': probe, 'neuron_id': clusters['cluster_id'],
        'region': clusters['region'], 'allen_acronym': clusters['acronym'],
        'x': clusters['x'], 'y': clusters['y'], 'z': clusters['z'],
        'p': p_value, 'r': r_value})))
        
    # Save to disk
    speed_df.to_csv(join(path_dict['save_path'], 'speed_neurons.csv'), index=False)
        
           