# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from itertools import combinations
from msvr_functions import paths, load_multiple_probes, load_objects, combine_regions
from koopman import NeuralKoopmanPipeline

# Settings
BIN_SIZE = 0.05
SMOOTHING = 0.025
T_BEFORE = 2
T_AFTER = 2

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
    
    # Loop over all brain region combinations
    for region_a, region_b in combinations(regions, 2):
        print(f'{region_a} <> {region_b}')
    
        # Get region spikes
        region_a_probe = region_probes[regions == region_a][0]
        region_a_neurons = clusters[region_a_probe]['cluster_id'][clusters[region_a_probe]['region'] == region_a]
        region_a_spikes = spikes[region_a_probe]['times'][np.isin(spikes[region_a_probe]['clusters'], region_a_neurons)]
        region_a_clusters = spikes[region_a_probe]['clusters'][np.isin(spikes[region_a_probe]['clusters'], region_a_neurons)]
        
        region_b_probe = region_probes[regions == region_b][0]
        region_b_neurons = clusters[region_b_probe]['cluster_id'][clusters[region_b_probe]['region'] == region_b]
        region_b_spikes = spikes[region_b_probe]['times'][np.isin(spikes[region_b_probe]['clusters'], region_b_neurons)]
        region_b_clusters = spikes[region_b_probe]['clusters'][np.isin(spikes[region_b_probe]['clusters'], region_b_neurons)]
                    
        # Create trials
        all_trials = []
        for trial_time in all_obj_df.loc[all_obj_df['object'] == 1, 'times']:
            
            these_spike_times, these_spike_ids = [], []
            
            # Area A
            these_spike_times.extend(region_a_spikes[
                (region_a_spikes > (trial_time - T_BEFORE)) & (region_a_spikes < (trial_time + T_AFTER))
                ])
            these_spike_ids.extend(region_a_clusters[
                (region_a_spikes > (trial_time - T_BEFORE)) & (region_a_spikes < (trial_time + T_AFTER))
                ]) 
                
            # Area B
            these_spike_times.extend(region_b_spikes[
                (region_b_spikes > (trial_time - T_BEFORE)) & (region_b_spikes < (trial_time + T_AFTER))
                ])
            these_spike_ids.extend(region_b_clusters[
                (region_b_spikes > (trial_time - T_BEFORE)) & (region_b_spikes < (trial_time + T_AFTER))
                ]) 
            
            all_trials.append((np.array(these_spike_times), np.array(these_spike_ids), trial_time - T_BEFORE))
        
        # Create neuron map
        area_map = {
            'A': list(np.unique(region_a_clusters)),
            'B': list(np.unique(region_b_clusters))
        }
        
        # Run Pipeline
        pipeline = NeuralKoopmanPipeline(dt=BIN_SIZE, sigma=SMOOTHING, n_delays=10)
        
        # Preprocess all trials
        trials_X_A, trials_X_B = pipeline.preprocess_trials(all_trials, area_map)
        
        # Fit the operators
        # This vertically stacks the Hankel matrices of all trials to learn one common dynamical law
        pipeline.fit_koopman_operators(trials_X_A, trials_X_B)
        
        # Analyze causality
        p_AB, p_BA, (null_c_AB, null_c_BA), (real_c_AB, real_c_BA) = pipeline.permutation_test(trials_X_A, trials_X_B)
        
        # Add to dataframe
        causality_df = pd.concat((causality_df, pd.DataFrame(data={
            'region_1': [region_a, region_b], 'region_2': [region_b, region_a],
            'region_pair': f'{region_a} <> {region_b}',
            'causality_score': [real_c_AB, real_c_BA], 'p_value': [p_AB, p_BA],
            'subject': subject, 'date': date
            })))

    # Save output
    causality_df.to_csv(path_dict['save_path'] / 'causality_koopman.csv', index=False)
        
       
