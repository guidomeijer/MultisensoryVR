# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:38:48 2025

By Guido Meijer
"""


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from msvr_functions import (paths, figure_style, load_neural_data, load_objects,
                            get_spike_counts_in_bins)

# Settings
BIN_SIZE = 10
D_BEFORE = 150  # mmn

# Create relative interval array
rel_intervals = np.vstack((np.arange(-D_BEFORE, 0, BIN_SIZE),
                           np.round(np.arange(-D_BEFORE + BIN_SIZE, 0 + BIN_SIZE, BIN_SIZE), 2))).T
bin_centers = np.arange(-D_BEFORE, 0, BIN_SIZE) + (BIN_SIZE / 2)

# Load in data
path_dict = paths()
neurons_df = pd.read_csv(path_dict['save_path'] / 'significant_neurons.csv')
neurons_df['date'] = neurons_df['date'].astype(str)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)

ramp_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\nStarting {subject} {date} {probe} [{i} of {rec.shape[0]}]\n')
    
    # Load in data
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
    spikes, clusters, channels = load_neural_data(session_path, probe, histology=True, only_good=True,
                                                  min_fr=0.1)
    trials = pd.read_csv(path_dict['local_data_path'] / 'Subjects' / subject / date / 'trials.csv')
    all_obj_df = load_objects(subject, date)
    
    # Get spike counts in bins before object 2
    spike_counts = []
    for t, this_pos in enumerate(all_obj_df.loc[all_obj_df['object'] == 2, 'distances']):
    
        these_intervals = rel_intervals + this_pos
        these_counts, these_ids = get_spike_counts_in_bins(spikes['distances'], spikes['clusters'],
                                                           these_intervals)
        spike_counts.append(these_counts)
    
    # Make into array
    spike_arr = np.dstack(spike_counts)  # neurons x bins x trials
    
    # Select significant neurons
    these_neurons = neurons_df[(neurons_df['date'] == date) & (neurons_df['probe'] == probe)]
    sig_neuron_ids = these_neurons.loc[these_neurons['p_context_obj2'] < 0.05, 'neuron_id'].values
    mean_arr = np.mean(spike_arr, axis=2)  # neurons x bins
    
    # Loop over neurons and do correlation
    p_values = np.empty(mean_arr.shape[0])
    for n in range(spike_arr.shape[0]):
        _, p_values[n] = stats.pearsonr(bin_centers, mean_arr[n, :])
        
    # Add to dataframe
    ramp_df = pd.concat((ramp_df, pd.DataFrame(data={
        'p_ramp': p_values, 'p_context': these_neurons['p_context_obj2'],
        'neuron_id': clusters['cluster_id'],
        'region': clusters['region'], 'acronym': clusters['acronym'],
        'subject': subject, 'date': date, 'probe': probe
        })))
    
# Save to disk
ramp_df.to_csv(path_dict['save_path'] / 'ramping_neurons.csv', index=False)