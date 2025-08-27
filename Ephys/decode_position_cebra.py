# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
import cebra
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from msvr_functions import (paths, load_neural_data, load_subjects, load_objects, bin_signal,
                            get_spike_counts_in_bins, figure_style)


# Settings
MIN_NEURONS = 5
BIN_SIZE = 0.025       # s
SPEED_THRESHOLD = 10 # mm/s
TIME_WINDOW = 0.250  # 250 ms for decoding
N_SHUFFLES = 10

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
colors, dpi = figure_style()

# Initialize CEBRA
cebra_pos_model = cebra.CEBRA(model_architecture='offset10-model',
                              batch_size=512,
                              learning_rate=3e-4,
                              temperature=1,
                              output_dimension=32,
                              max_iterations=500,
                              distance='cosine',
                              conditional='time_delta',
                              device='cuda',
                              verbose=True,
                              time_offsets=10)
pos_decoder = cebra.KNNDecoder(n_neighbors=36, metric='cosine')

# %% Loop over recordings

error_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\n{subject} {date} {probe} ({i} of {rec.shape[0]})')

    # Load in data
    session_path = path_dict['local_data_path'] / 'Subjects' / subject / date
    spikes, clusters, channels = load_neural_data(session_path, probe)
    trials = pd.read_csv(session_path / 'trials.csv')
    all_obj_df = load_objects(subject, date)
    animal_position = np.load(session_path / 'continuous.wheelDistance.npy')
    animal_speed = np.load(session_path / 'continuous.wheelSpeed.npy')
    timestamps = np.load(session_path / 'continuous.times.npy')
    
    # Reformat distance into relative distance in the environment
    rel_dist_list, intervals_list, speed_list = [], [], []
    for i in trials.index.values[1:]:
        
        # Get relative distance in this trial
        trial_ind = ((timestamps >= trials.loc[i, 'enterEnvTime'])
                     & (timestamps <= trials.loc[i, 'exitEnvTime']))
        rel_dist = animal_position[trial_ind] - trials.loc[i, 'enterEnvPos']
        this_time = timestamps[trial_ind]   
        this_speed = animal_speed[trial_ind]
        
        # Bin distance and speed
        bin_edges = np.arange(this_time[0], this_time[-1], BIN_SIZE)
        trial_intervals = np.vstack((bin_edges[:-1], bin_edges[1:])).T
        binned_dist = bin_signal(this_time, rel_dist, bin_edges)
        binned_speed = bin_signal(this_time, this_speed, bin_edges)
                        
        # Add to lists
        rel_dist_list.append(binned_dist)
        speed_list.append(binned_speed)
        intervals_list.append(trial_intervals)
        
    # Convert lists to np arrays
    rel_distance = np.concatenate(rel_dist_list)
    intervals = np.vstack(intervals_list)
    speed = np.concatenate(speed_list)

    # Apply speed threshold
    rel_distance = rel_distance[speed >= SPEED_THRESHOLD]
    intervals = intervals[speed >= SPEED_THRESHOLD, :]
            
    # %% Loop over regions
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        print(f'\nStarting {region}')
        
        # Get region neurons
        use_neurons = clusters['cluster_id'][clusters['region'] == region]
        if use_neurons.shape[0] < MIN_NEURONS:
            continue
        region_spikes = spikes['times'][np.isin(spikes['clusters'], use_neurons)]
        region_clusters = spikes['clusters'][np.isin(spikes['clusters'], use_neurons)]
        
        # Get spike counts
        spike_counts, neuron_ids = get_spike_counts_in_bins(region_spikes, region_clusters, intervals)
        spike_counts = spike_counts.T  # timebins x neurons
        
        # Fit model
        cebra_pos_model.fit(spike_counts, rel_distance)
        
        # Decode position
        embedding = cebra_pos_model.transform(spike_counts)
        pos_decoder.fit(embedding, rel_distance)
        pos_pred = pos_decoder.predict(embedding)
        error = np.mean(np.abs(rel_distance - pos_pred))
        
        # Decode shuffled
        shuf_distance = shuffle(rel_distance)
        cebra_pos_model.fit(spike_counts, shuf_distance)
        embedding = cebra_pos_model.transform(spike_counts)
        pos_decoder.fit(embedding, shuf_distance)
        pos_pred_shuf = pos_decoder.predict(embedding)
        error_shuf = np.mean(np.abs(rel_distance - pos_pred_shuf))
        
        # Add to dataframe
        error_df = pd.concat((error_df, pd.DataFrame(index=[error_df.shape[0]], data={
            'error_mm': error, 'error_mm_shuffled': error_shuf, 'region': region,
            'subject': subject, 'date': date,
            'n_neurons': use_neurons.shape[0], 'n_trials': trials.shape[0]})))
        
        # Plot result
        f, ax1 = plt.subplots(1, 1, figsize=(3, 2), dpi=dpi)
        ax1.plot(rel_distance, label='Actual position')
        ax1.plot(pos_pred, label='Decoded position')
        ax1.set(ylabel='Position in environment (mm)', xlim=[0, 2000], xticks=[])
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(path_dict['fig_path'] / 'DecodePosition' / f'{region}_{subject}_{date}.jpg', dpi=600)
        plt.close(f)
    
    # Save results
    error_df.to_csv(path_dict['save_path'] / 'position_decoding_error.csv', index=False)
    
       