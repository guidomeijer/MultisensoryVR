#%%
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
from scipy.ndimage import gaussian_filter
from msvr_functions import (paths, load_neural_data, load_objects, combine_regions,
                            to_spikeship_dataformat, figure_style)
colors, dpi = figure_style()

# Settings
T_BEFORE = 1.5  # s
T_AFTER = 2.5
BIN_SIZE = 0.4
STEP_SIZE = 0.05
MIN_NEURONS = 5
MIN_RIPPLES = 20
N_CPUS = 20
MIN_SPIKES_PER_BIN = 5
SMOOTHING_SIGMA = 1
CA1_COMPR = 8
RIPPLE_DELAY = 0.75

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
def run_spikeship(this_bin_center, use_spikes, use_clusters, event_times, use_ripple_times, min_spikes, compression):
        
    # Get event spike epochs
    event_intervals = np.vstack((event_times + (this_bin_center - (BIN_SIZE/2)),
                                 event_times + (this_bin_center + (BIN_SIZE/2)))).T
    ss_events, ii_events, n_spikes_events = to_spikeship_dataformat(
        use_spikes, use_clusters, event_intervals, min_spikes=min_spikes)

    # Get ripple spike epochs with compression factor
    ripple_intervals = np.vstack((use_ripple_times - (BIN_SIZE/2),
                                  use_ripple_times + (BIN_SIZE/2))).T
    ss_ripples, ii_ripples, n_spikes_ripples = to_spikeship_dataformat(
        use_spikes, use_clusters, ripple_intervals, min_spikes=min_spikes, compression_factor=compression)

    # Concatenate together
    ss_spike_times = np.concatenate((ss_events, ss_ripples))
    ii_spike_times = np.vstack((ii_events, ii_ripples + ss_events.shape[0]))

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
    ripple_times = these_ripples['start_times'] + ((these_ripples['end_times'] - these_ripples['start_times']) / 2)
    if ripple_times.shape[0] < MIN_RIPPLES:
        continue

    # Loop over regions
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        print(f'Starting {region}')

        # Set temporal compression factor
        if region == 'CA1':
            compression_factor = CA1_COMPR
            this_delay = 0
        elif region == 'VIS':
            compression_factor = 1
            this_delay = RIPPLE_DELAY
        else:
            compression_factor = 1
            this_delay = 0

        # Get region neurons
        region_neurons = clusters['cluster_id'][clusters['region'] == region]
        region_spikes = spikes['times'][np.isin(spikes['clusters'], region_neurons)]
        region_clusters = spikes['clusters'][np.isin(spikes['clusters'], region_neurons)]
        if np.unique(region_clusters).shape[0] < MIN_NEURONS:
            continue

        # Loop over rewarded objects
        for obj in [1, 2]:

            # Run SpikeShip on goal entries
            goal_times = all_obj_df.loc[(all_obj_df['object'] == obj) & (all_obj_df['goal'] == 1), 'times'].values
            goal_results = Parallel(n_jobs=N_CPUS)(
                delayed(run_spikeship)(bin_center, region_spikes, region_clusters, goal_times,
                                       ripple_times + this_delay, min_spikes=MIN_SPIKES_PER_BIN,
                                       compression=compression_factor)
                for bin_center in t_centers)
            goal_diss = np.array([clean_spikeship_nans(i) for i in goal_results])

            # Calculate contrast metric
            within_a = goal_diss[:, :goal_times.shape[0], :goal_times.shape[0]]
            within_b = goal_diss[:, goal_times.shape[0]:, goal_times.shape[0]:]
            between_block = goal_diss[:, :goal_times.shape[0], goal_times.shape[0]:]
            goal_contrast = np.mean(between_block, axis=(1, 2)) - (mean_no_diag(within_a) + mean_no_diag(within_b)) / 2
            goal_contrast = gaussian_filter(goal_contrast, SMOOTHING_SIGMA)
            goal_contrast_bl = goal_contrast - np.mean(goal_contrast[t_centers < -0.5])

            # Add to dataframe
            spikeship_df = pd.concat((spikeship_df, pd.DataFrame(data={
                'contrast': goal_contrast, 'contrast_bl': goal_contrast_bl, 'goal': 1, 'time': t_centers,
                'object': obj, 'region': region, 'subject': subject, 'date': date, 'probe': probe
                })))

            # No goal entries
            no_goal_times = all_obj_df.loc[(all_obj_df['object'] == obj) & (all_obj_df['goal'] == 0), 'times'].values
            no_goal_results = Parallel(n_jobs=N_CPUS)(
                delayed(run_spikeship)(bin_center, region_spikes, region_clusters, no_goal_times,
                                       ripple_times + this_delay, min_spikes=MIN_SPIKES_PER_BIN,
                                       compression=compression_factor)
                for bin_center in t_centers)
            no_goal_diss = np.array([clean_spikeship_nans(i) for i in no_goal_results])

            # Calculate contrast metric
            within_a = no_goal_diss[:, :no_goal_times.shape[0], :no_goal_times.shape[0]]
            within_b = no_goal_diss[:, no_goal_times.shape[0]:, no_goal_times.shape[0]:]
            between_block = no_goal_diss[:, :no_goal_times.shape[0], no_goal_times.shape[0]:]
            no_goal_contrast = np.mean(between_block, axis=(1, 2)) - (mean_no_diag(within_a) + mean_no_diag(within_b)) / 2
            no_goal_contrast = gaussian_filter(no_goal_contrast, SMOOTHING_SIGMA)
            no_goal_contrast_bl = no_goal_contrast - np.mean(no_goal_contrast[t_centers < -0.5])

            # Add to dataframe
            spikeship_df = pd.concat((spikeship_df, pd.DataFrame(data={
                'contrast': no_goal_contrast, 'contrast_bl': no_goal_contrast_bl, 'goal': 0, 'time': t_centers,
                'object': obj, 'region': region, 'subject': subject, 'date': date, 'probe': probe
                })))

        # Process non-rewarded object 3
        obj3_times = all_obj_df.loc[all_obj_df['object'] == 3, 'times'].values
        goal_results = Parallel(n_jobs=N_CPUS)(
            delayed(run_spikeship)(bin_center, region_spikes, region_clusters, obj3_times,
                                   ripple_times + this_delay, min_spikes=MIN_SPIKES_PER_BIN,
                                   compression=compression_factor)
            for bin_center in t_centers)
        obj3_diss = np.array([clean_spikeship_nans(i) for i in goal_results])

        # Calculate contrast metric
        within_a = obj3_diss[:, :obj3_times.shape[0], :obj3_times.shape[0]]
        within_b = obj3_diss[:, obj3_times.shape[0]:, obj3_times.shape[0]:]
        between_block = obj3_diss[:, :obj3_times.shape[0], obj3_times.shape[0]:]
        obj3_contrast = np.mean(between_block, axis=(1, 2)) - (mean_no_diag(within_a) + mean_no_diag(within_b)) / 2
        obj3_contrast = gaussian_filter(obj3_contrast, SMOOTHING_SIGMA)
        obj3_contrast_bl = obj3_contrast - np.mean(obj3_contrast[t_centers < -0.5])

        # Add to dataframe
        spikeship_df = pd.concat((spikeship_df, pd.DataFrame(data={
            'contrast': obj3_contrast, 'contrast_bl': obj3_contrast_bl, 'goal': 0, 'time': t_centers,
            'object': 3, 'region': region, 'subject': subject, 'date': date, 'probe': probe
        })))

# Save to disk
spikeship_df.to_csv(path_dict['google_drive_data_path'] / f'spikeship_ripples_{RIPPLE_DELAY}s_{CA1_COMPR}x.csv',
                    index=False)
