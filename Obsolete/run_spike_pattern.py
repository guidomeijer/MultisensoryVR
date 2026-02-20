# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy import stats, signal
import matplotlib.pyplot as plt
from zetapy import zetatstest
from msvr_functions import (paths, load_multiple_probes, load_objects, combine_regions,
                            peri_event_trace, figure_style, N_PATTERNS, plot_ppseq)
from ppseq.model import PPSeq
colors, dpi = figure_style()

# Settings
MIN_NEURONS = 5
BIN_WIDTH = 0.01
NBINS_PATTERNS = 15
SIG_TIME = 1  # s
OVERWRITE = True
PLOT = False
MIN_RIPPLES = 20
RIPPLE_WIN = 0.4  # s

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)

if OVERWRITE:
    pattern_df = pd.DataFrame()
else:
    pattern_df = pd.read_csv(path_dict['save_path'] / 'spike_pattern_sig.csv')
    pattern_df['subject'] = pattern_df['subject'].astype(str)
    pattern_df['date'] = pattern_df['date'].astype(str)
    pattern_keys = pattern_df[['subject', 'date']].drop_duplicates()
    rec_merged = rec.merge(pattern_keys, on=['subject', 'date'], how='left', indicator=True)
    rec = rec_merged[rec_merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    rec = rec.reset_index(drop=True)

#%% Functions
def get_windowed_means(event_times, time_ax, signal, window=(-1, 0)):
    """
    Extracts the mean signal amplitude within a specific window relative to event times.
    """
    event_means = []
    for t in event_times:
        # Define the absolute time boundaries
        t_start = t + window[0]
        t_end = t + window[1]
        
        # Find indices within the time window
        mask = (time_ax >= t_start) & (time_ax < t_end)
        
        if np.any(mask):
            event_means.append(np.mean(signal[mask]))
        else:
            # Handle cases where the window might be outside the recorded time_ax
            event_means.append(np.nan)
            
    return np.array(event_means)



#%% MAIN
for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'\nStarting {subject} {date} [{i} of {rec.shape[0]}]')

    # Load in data
    session_path = path_dict['local_data_path'] / 'subjects' / f'{subject}' / f'{date}'
    trials = pd.read_csv(session_path / 'trials.csv')
    spikes, clusters, channels = load_multiple_probes(session_path, min_fr=0.5)
    all_obj_df = load_objects(subject, date)

    # Merge regions
    for probe in clusters.keys():
        clusters[probe]['region'] = combine_regions(clusters[probe]['acronym'], split_peri=False)

    # Get list of all regions and which probe they were recorded on
    regions, region_probes = [], []
    for p, probe in enumerate(spikes.keys()):
        unique_regions = np.unique(clusters[probe]['region'])
        unique_regions = unique_regions[unique_regions != 'root']
        if (probe == 'probe00') & ('CA1' in unique_regions):
            unique_regions = unique_regions[unique_regions != 'CA1']
        regions.append(unique_regions)
        region_probes.append([probe] * unique_regions.shape[0])
    regions = np.concatenate(regions)
    region_probes = np.concatenate(region_probes)

    # Loop over all brain regions
    for region, probe in zip(regions, region_probes):
        print(f'{region}')

        # Get spiking activity from this region
        region_neuron_ids = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
        if (region_neuron_ids.shape[0] < MIN_NEURONS) or (region == 'root'):
            continue
        region_spikes = spikes[probe]['times'][np.isin(spikes[probe]['clusters'], region_neuron_ids)]
        region_clusters = spikes[probe]['clusters'][np.isin(spikes[probe]['clusters'], region_neuron_ids)]

        # Create input for PPSeq
        list_of_spiketimes = []
        for n, neuron_id in enumerate(region_neuron_ids):
            list_of_spiketimes.append(region_spikes[region_clusters == neuron_id])
        num_timesteps = int(region_spikes.max() // BIN_WIDTH) + 1
        time_ax = np.arange(num_timesteps) * BIN_WIDTH + BIN_WIDTH / 2
        num_neurons = len(list_of_spiketimes)

        indices = []
        values = []
        for ii, these_spike_times in enumerate(list_of_spiketimes):
            bins = (torch.tensor(these_spike_times, device='cuda') // BIN_WIDTH).long()
            indices.append(torch.stack([torch.full_like(bins, ii), bins]))
            values.append(torch.ones_like(bins, dtype=torch.float32))

        if indices:
            indices = torch.cat(indices, dim=1)
            values = torch.cat(values)
            data = torch.sparse_coo_tensor(indices, values, size=(num_neurons, num_timesteps),
                                           device='cuda').to_dense()
        else:
            data = torch.zeros(num_neurons, num_timesteps, device='cuda')
        data = data[~torch.all(data == 0, dim=1)]

        # Run ppseq
        model = PPSeq(num_templates=N_PATTERNS[region],
                      num_neurons=num_neurons,
                      template_duration=NBINS_PATTERNS,
                      alpha_a0=1.5,
                      beta_a0=1,
                      alpha_b0=1,
                      beta_b0=0.1,
                      alpha_t0=1.2,
                      beta_t0=0.1)
        lps, amplitudes_torch = model.fit(data, num_iter=200)
        amplitudes = amplitudes_torch.cpu().numpy()
        templates = model.templates.cpu()

        # Convolve the spike pattern events with the pattern's temporal profile
        activation_rate = np.zeros_like(amplitudes)
        for pattern_idx in range(amplitudes.shape[0]):
            
            # Summing across neurons collapses the spatial dimension, leaving just the time course
            pattern_profile = templates[pattern_idx].sum(axis=0) 
            
            # 2. Reconstruct the Continuous Activation Trace (Convolution)
            # This converts sparse impulses into the actual shape of the activity
            amp_series = amplitudes[pattern_idx, :]
            continuous_rate = signal.convolve(amp_series, pattern_profile, mode='full')
            activation_rate[pattern_idx, :] = continuous_rate[:len(amp_series)]
            

        if PLOT:
            # Plot patterns
            f, ax = plot_ppseq(data.cpu()[:, -5000:-4000], model, amplitudes_torch.cpu()[:, -5000:-4000])
            ax.set(xticks=[0, 33, 66, 100], xticklabels=[0, 5, 10, 15], yticks=[],
                   ylabel='Sorted neurons', xlabel='Time (s)')
            sns.despine(trim=True, left=True)
            plt.tight_layout()
            plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns' / f'{region}_{subject}_{date}_example.jpg', dpi=600)
            plt.close(f)

        
        # Save spike pattern amplitudes to disk
        np.save(path_dict['google_drive_data_path'] / 'SpikePatterns' / f'{subject}_{date}_{region}.amplitudes.npy',
                amplitudes)
        np.save(path_dict['google_drive_data_path'] / 'SpikePatterns' / f'{subject}_{date}_{region}.times.npy',
                time_ax)
        

