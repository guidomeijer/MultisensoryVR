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
BIN_WIDTH = 0.01
NBINS_PATTERNS = 15
N_PATTERNS = 12

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])

spike_pattern_n_df = pd.DataFrame()
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
        
        # Train on first half, Test on second half
        T = data.shape[1]
        split_idx = T // 2
        train_data = data[:, :split_idx]
        test_data = data[:, split_idx:]
        
        # Ensure data is on GPU
        train_data = train_data.cuda()
        test_data = test_data.cuda()

        # Initialize storage
        train_lls = np.empty(N_PATTERNS)
        test_lls = np.empty(N_PATTERNS)
        
        # --- 3. Cross-Validation Loop ---
        for pp, n_patterns in enumerate(np.arange(1, N_PATTERNS+1)):
            
            # --- STEP A: TRAIN (Learn Templates) ---
            model_train = PPSeq(num_templates=n_patterns,
                                num_neurons=num_neurons,
                                template_duration=NBINS_PATTERNS,
                                alpha_a0=1.5,
                                beta_a0=1,
                                alpha_b0=1,
                                beta_b0=0.1,
                                alpha_t0=1.2,
                                beta_t0=0.1
                                )
            
            # Fit fully on training data
            _, amplitudes = model_train.fit(train_data, num_iter=200)
            
            # Save Train Score
            # We must infer amplitudes one last time to get the strict LL
            ll_train = model_train.log_likelihood(train_data, amplitudes).cpu().numpy()
            train_lls[pp] = ll_train.item() / train_data.numel() # Normalize by data size

            # --- STEP B: TEST (Evaluate Generalization) ---
            # 1. Initialize a new model for the test set
            model_test = PPSeq(num_templates=n_patterns,
                               num_neurons=num_neurons,
                               template_duration=NBINS_PATTERNS,
                               alpha_a0=1.5,
                               beta_a0=1,
                               alpha_b0=1,
                               beta_b0=0.1,
                               alpha_t0=1.2,
                               beta_t0=0.1
                               )
            
            # 2. TRANSFER KNOWLEDGE: Copy trained templates to test model
            # We assume model.templates is a torch.nn.Parameter
            with torch.no_grad():
                model_test.templates.data.copy_(model_train.templates.data)
                # Also copy background rates if your model learns them
                if hasattr(model_train, 'background_rate'):
                    model_test.background_rate.data.copy_(model_train.background_rate.data)

            # 3. FREEZE TEMPLATES: Prevent the model from "cheating" by learning new shapes
            model_test.templates.requires_grad = False
            
            # 4. INFERENCE ONLY: Fit only the Amplitudes on Test Data
            # Since templates are frozen (requires_grad=False), the optimizer 
            # will only update the amplitude latents to explain the test spikes.
            lps_test, amps_test = model_test.fit(test_data, num_iter=200)
            
            # 5. Calculate Test Score
            ll_test = model_test.log_likelihood(test_data, amps_test)
            test_lls[pp] = ll_test.item() / test_data.numel() # Normalize by data size
            
            print(f"Patterns: {n_patterns} | Train LL: {train_lls[pp]:.4f} | Test LL: {test_lls[pp]:.4f}")
            
        # --- 4. Save Results ---
        spike_pattern_n_df = pd.concat((spike_pattern_n_df, pd.DataFrame(data={
            'n_patterns': np.arange(1, N_PATTERNS+1),
            'train_log_likelihood': train_lls,
            'test_log_likelihood': test_lls,
            'region': region,
            'subject': subject, 
            'date': date
            })))
        
    # Save to disk
    spike_pattern_n_df.to_csv(path_dict['save_path'] / 'spike_pattern_n_select.csv', index=False)