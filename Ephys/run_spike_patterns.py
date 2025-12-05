# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import torch
from msvr_functions import paths, load_multiple_probes, load_objects, combine_regions
from ppseq.plotting import plot_model, color_plot
from ppseq.model import PPSeq

# Settings
MIN_NEURONS = 5
BIN_WIDTH = 0.1
N_PATTERNS = 12

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])

causality_df = pd.DataFrame()
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
        
        # Select spikes in active period
        region_clusters = region_clusters[((region_spikes > trials['enterEnvTime'].values[0])
                                           & (region_spikes < trials['exitEnvTime'].values[-1]))]
        region_spikes = region_spikes[((region_spikes > trials['enterEnvTime'].values[0])
                                       & (region_spikes < trials['exitEnvTime'].values[-1]))]
        
        # Create input for PPSeq
        list_of_spiketimes = []
        for n, neuron_id in enumerate(region_neuron_ids):
            list_of_spiketimes.append(region_spikes[region_clusters == neuron_id])
        num_timesteps = int(region_spikes.max() // BIN_WIDTH) + 1
        time_ax = np.arange(0, region_spikes.max(), BIN_WIDTH) + BIN_WIDTH / 2
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
        
        
        # Fit PPSeq model with increasing number of patterns
        log_likelihood = np.empty(N_PATTERNS - 1)
        for pp, n_patterns in enumerate(range(1, N_PATTERNS)):
            model = PPSeq(num_templates=n_patterns,
                          num_neurons=num_neurons,
                          template_duration=10,
                          alpha_a0=1.5,
                          beta_a0=0.2,
                          alpha_b0=1,
                          beta_b0=0.1,
                          alpha_t0=1.2,
                          beta_t0=0.1)
            lps, amplitudes = model.fit(data, num_iter=100)
            log_likelihood[pp] = model.log_likelihood(data, amplitudes).cpu().numpy()
        
        asd
      
        
       
