# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from elephant.gpfa import GPFA
import neo
import quantities as pq
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from msvr_functions import (paths, load_multiple_probes, load_subjects, load_objects,
                            calculate_peths, figure_style)
colors, dpi = figure_style()

# Settings
T_BEFORE = 2  # s
T_AFTER = 1
BIN_SIZE = 0.05
MIN_NEURONS = 10
MIN_TRIALS = 30
N_CORES = 6
N_DIMS = 10
MAX_LAG = 0.5  # s
MIN_SPIKES_PER_TRIAL = 2
SUBJECT = '459601'
SESSION = '20240411'

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))

# Load in data
session_path = join(path_dict['local_data_path'], 'subjects', f'{SUBJECT}', f'{SESSION}')
spikes, clusters, channels = load_multiple_probes(session_path, min_fr=0.5)
trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', SUBJECT, SESSION, 'trials.csv'))
all_obj_df = load_objects(SUBJECT, SESSION)

def create_spiketrains(spike_times, spike_ids, selected_neuron_ids, trial_start_time, trial_duration_s):
    """
    Creates a list of neo.SpikeTrain objects for a single trial from raw spike data,
    using only the neurons specified in selected_neuron_ids.

    Args:
        spike_times (np.ndarray): Array of all spike times.
        spike_ids (np.ndarray): Array of neuron IDs for each spike.
        selected_neuron_ids (list or np.ndarray): A list of neuron IDs to include.
        trial_start_time (float): The start time of the trial in seconds.
        trial_duration_s (float): The duration of the trial in seconds.

    Returns:
        list: A list of neo.SpikeTrain objects.
    """
    trial_spiketrains = []
    trial_end_time = trial_start_time + trial_duration_s

    # Iterate only through the selected neuron IDs
    for neuron_id in selected_neuron_ids:
        # 1. Filter all spikes to get just those from the current neuron
        neuron_mask = (spike_ids == neuron_id)
        neuron_spike_times = spike_times[neuron_mask]

        # 2. Filter the neuron's spikes to get just those in the current trial
        trial_mask = (neuron_spike_times >= trial_start_time) & (neuron_spike_times < trial_end_time)
        spikes_in_trial = neuron_spike_times[trial_mask]

        # 3. Make spike times relative to the start of the trial
        relative_spike_times = spikes_in_trial - trial_start_time

        # 4. Create the neo.SpikeTrain object with correct units and duration
        st = neo.SpikeTrain(relative_spike_times * pq.s, t_stop=trial_duration_s * pq.s)
        trial_spiketrains.append(st)

    return trial_spiketrains

# %% Loop over regions

# Get list of all regions and which probe they were recorded on
regions, region_probes = [], []
for p, probe in enumerate(spikes.keys()):
    regions.append(np.unique(clusters[probe]['region']))
    region_probes.append([probe] * np.unique(clusters[probe]['region']).shape[0])
regions = np.concatenate(regions)
region_probes = np.concatenate(region_probes)

prob, residuals = dict(), dict()
for r, (region, probe) in enumerate(zip(regions, region_probes)):
    if region == 'root':
        continue
    print(f'Decoding {region}')
    
    # Get region neurons
    region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
    if region_neurons.shape[0] < MIN_NEURONS:
        continue

    # Get lists of spike trains in neo format
    print('Creating lists of spike times in neo format...')
    trials_spiketrains = []
    for t, object_entry in enumerate(all_obj_df['times'].values):
        trial_spiketrains = create_spiketrains(
            spikes[probe]['times'], spikes[probe]['clusters'], region_neurons,
            object_entry - T_BEFORE, T_BEFORE + T_AFTER)
        #trial_spiketrains = [i for i in trial_spiketrains if len(i) >= MIN_SPIKES_PER_TRIAL]
        trials_spiketrains.append(trial_spiketrains)

    # Fit GPFA
    gpfa = GPFA(bin_size=BIN_SIZE * pq.s, x_dim=N_DIMS)
    latent_variables = gpfa.fit_transform(trials_spiketrains)

    # Plot
    obj1_ind = all_obj_df[all_obj_df['object'] == 1].index.values
    obj2_ind = all_obj_df[all_obj_df['object'] == 2].index.values
    obj3_ind = all_obj_df[all_obj_df['object'] == 3].index.values

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.75*2, 1.75), dpi=dpi)
    for trial_ind in obj1_ind:
        ax1.plot(latent_variables[trial_ind][0, :], latent_variables[trial_ind][1, :], '-', lw=0.5, c='C0', alpha=0.5)
    average_trajectory = np.mean(latent_variables[obj1_ind], axis=0)
    ax1.plot(average_trajectory[0], average_trajectory[1], '-', lw=2, c='C1')

    for trial_ind in obj2_ind:
        ax2.plot(latent_variables[trial_ind][0, :], latent_variables[trial_ind][1, :], '-', lw=0.5, c='C0', alpha=0.5)
    average_trajectory = np.mean(latent_variables[obj2_ind], axis=0)
    ax2.plot(average_trajectory[0], average_trajectory[1], '-', lw=2, c='C1')

    plt.show()

