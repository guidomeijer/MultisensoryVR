# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from spikeship import spikeship
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from msvr_functions import (paths, load_neural_data, load_objects, combine_regions,
                            to_spikeship_dataformat, figure_style)
colors, dpi = figure_style()

# Settings
T_BEFORE = 2  # s
T_AFTER = 2
BIN_SIZE = 0.3
STEP_SIZE = 0.05
MIN_NEURONS = 5
MIN_RIPPLES = 20
N_CPUS = 18
MIN_SPIKES_PER_BIN = 10

# Create time array
t_centers = np.arange(-T_BEFORE + (BIN_SIZE/2), T_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)

#%% FUNCTIONS
def run_spikeship(this_bin_center, use_spikes, use_clusters, event_times, ripple_times, min_spikes):
        
    # Get time intervals
    event_intervals = np.vstack((event_times + (this_bin_center - (BIN_SIZE/2)),
                                 event_times + (this_bin_center + (BIN_SIZE/2)))).T
    ripple_intervals = np.vstack((ripple_times - (BIN_SIZE/2),
                                  ripple_times + (BIN_SIZE/2))).T
    use_intervals = np.vstack((event_intervals, ripple_intervals))
    
    # Transform to spikeship data format
    ss_spike_times, ii_spike_times, n_spikes = to_spikeship_dataformat(
        use_spikes, use_clusters, use_intervals, min_spikes=min_spikes)
    
    # Run SpikeShip
    diss_spikeship = spikeship.distances(ss_spike_times, ii_spike_times)
    
    return diss_spikeship

def clean_spikeship_nans(diss_arr):
    D = diss_arr.copy()
    is_silent = np.isnan(D).all(axis=1)
    
    # Use 95th percentile instead of Max to avoid outlier stretching
    if np.all(np.isnan(D)): return np.zeros_like(D)
    
    # We use nanpercentile to find a 'reasonable' maximum distance
    ceiling_val = np.nanpercentile(D, 95) 
    
    # Fill NaNs with this ceiling
    D[np.isnan(D)] = ceiling_val
    np.fill_diagonal(D, 0)
    
    # Silent vs Silent should be 0 (they are identically empty)
    silent_mask = np.outer(is_silent, is_silent)
    D[silent_mask] = 0.0
    
    return D

def mean_no_diag(arr_3d):
    # This masks the diagonal of the last two dimensions
    mask = ~np.eye(arr_3d.shape[1], dtype=bool)
    return np.mean(arr_3d[:, mask], axis=1)

#%% MAIN

spikeship_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\n{subject} {date} {probe} ({i} of {rec.shape[0]})')

    # Load in data
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
    spikes, clusters, channels = load_neural_data(session_path, probe)
    trials = pd.read_csv(path_dict['local_data_path'] / 'Subjects' / subject / date / 'trials.csv')
    all_obj_df = load_objects(subject, date)    
    these_ripples = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
    ripple_centers = (these_ripples['start_times'] + ((these_ripples['end_times'] - these_ripples['start_times']) / 2)).values
    if ripple_centers.shape[0] < MIN_RIPPLES:
        continue    
        
    # %% Loop over regions
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        print(f'Starting {region}')
        
        # Get region neurons
        region_neurons = clusters['cluster_id'][clusters['region'] == region]
        region_spikes = spikes['times'][np.isin(spikes['clusters'], region_neurons)]
        region_clusters = spikes['clusters'][np.isin(spikes['clusters'], region_neurons)]
        if np.unique(region_clusters).shape[0] < MIN_NEURONS:
            continue

        # Loop over objects
        for obj in [1, 2]:
            
            # Run SpikeShip in parallel over timebins
            trial_times = all_obj_df.loc[all_obj_df['object'] == obj, 'times'].values
            results = Parallel(n_jobs=N_CPUS)(
                delayed(run_spikeship)(bin_center, region_spikes, region_clusters, trial_times,
                                       ripple_centers, min_spikes=MIN_SPIKES_PER_BIN)
                for bin_center in t_centers)
            
            # Clean up NaNs (trials with too few spikes)
            clean_results = [clean_spikeship_nans(i) for i in results]
            diss_arr_3d = np.array(clean_results)
            
            # Calculate similarity metrics
            within_a = diss_arr_3d[:, :trial_times.shape[0], :trial_times.shape[0]]
            within_b = diss_arr_3d[:, trial_times.shape[0]:, trial_times.shape[0]:]
            between_block = diss_arr_3d[:, :trial_times.shape[0], trial_times.shape[0]:]
            contrast_metric = np.mean(between_block, axis=(1, 2)) - (mean_no_diag(within_a) + mean_no_diag(within_b)) / 2
            similarity_metric = np.median(between_block, axis=(1, 2))
            
            # Baseline subtract
            contrast_bl = contrast_metric - np.mean(contrast_metric[t_centers < 0])
            similarity_bl = similarity_metric - np.mean(similarity_metric[t_centers < 0])
            
            # Add to dataframe
            spikeship_df = pd.concat((spikeship_df, pd.DataFrame(data={
                'similarity': similarity_metric, 'contrast': contrast_metric, 'time': t_centers,
                'similarity_bl': similarity_bl, 'contrast_bl': contrast_bl,
                'object': obj, 'region': region, 'subject': subject, 'date': date, 'probe': probe
                })))

# %% Plot

f, axs = plt.subplots(1, 6, figsize=(7, 2), dpi=dpi, sharey=False, sharex=True)
axs = axs.flatten()
plot_df = spikeship_df[spikeship_df['object'] == 2]
for i, region in enumerate(plot_df['region'].unique()):
    axs[i].plot([-1, 2], [0, 0], lw=0.5, ls='--')
    sns.lineplot(data=plot_df[plot_df['region'] == region], x='time', y='contrast_bl',
                 ax=axs[i], errorbar='se', err_kws={'lw': 0})
    axs[i].set(title=region, xlim=[-1, 2])
sns.despine(trim=True)
plt.tight_layout()