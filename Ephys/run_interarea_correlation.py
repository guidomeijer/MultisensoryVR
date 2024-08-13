# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, sem
from msvr_functions import paths, load_neural_data, load_subjects, calculate_peths, load_objects

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
T_BEFORE = 1
T_AFTER = 2
BIN_SIZE = 0.1
SMOOTHING = 0.15
SUBTRACT_MEAN = True
MIN_NEURONS = 10  # per region

# Initialize
path_dict = paths()
subjects = load_subjects()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
all_obj_df = load_objects(SUBJECT, DATE)
rew_obj2_df = all_obj_df[all_obj_df['object'] == 2]

# %% Fuction for parallization

def run_correlation(spike_counts, tt):
    pairwise_corr = []
    for n1 in range(spike_counts[region_1].shape[1]):  # Neurons in region 1
        for n2 in range(spike_counts[region_2].shape[1]):  # Neurons in region 2
            r, _ = pearsonr(spike_counts[region_1][:, n1, tt],
                            spike_counts[region_2][:, n2, tt])
            pairwise_corr.append(r)
    r_mean = np.nanmean(pairwise_corr)
    r_sem = sem(pairwise_corr, nan_policy='omit')
  
    return r_mean, r_sem


# %%
# Get binned spike counts per region
goal_counts, distractor_counts, sound_counts = dict(), dict(), dict()
corr_df = pd.DataFrame()
for j, region in enumerate(np.unique(clusters['region'])):
    if region == 'root':
        continue
    
    # Select neurons
    region_neurons = clusters['cluster_id'][clusters['region'] == region]
    #sig_neurons = neurons_df.loc[neurons_df['sig_goal'] | neurons_df['sig_obj_onset'], 'neuron_id'].values
    sig_neurons = neurons_df.loc[neurons_df['sig_obj_onset'], 'neuron_id'].values
    #sig_neurons = neurons_df.loc[neurons_df['sig_goal'], 'neuron_id'].values
    use_neurons = region_neurons[np.isin(region_neurons, sig_neurons)]
    if use_neurons.shape[0] < MIN_NEURONS:
        continue

    # Get spike counts
    peths, goal_counts[region] = calculate_peths(
        spikes['times'], spikes['clusters'], use_neurons,
        rew_obj2_df.loc[rew_obj2_df['goal'] == 1, 'times'], T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING,
        return_fr=False)
    
    peths, distractor_counts[region] = calculate_peths(
        spikes['times'], spikes['clusters'], use_neurons,
        rew_obj2_df.loc[rew_obj2_df['goal'] == 0, 'times'], T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING,
        return_fr=False)
    
    peths, sound_counts[region] = calculate_peths(
        spikes['times'], spikes['clusters'], use_neurons,
        trials['soundOnset'].values, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING,
        return_fr=False)

    # Get time scale
    tscale = peths['tscale']

# Get pairwise neural correlations between all neuron pairs in both regions
these_regions = list(goal_counts.keys())
for r1, region_1 in enumerate(these_regions[:-1]):
    for r2, region_2 in enumerate(these_regions[r1+1:]):
        print(f'{region_1} - {region_2}')
        
        # Goal object entries
        results = Parallel(n_jobs=-1)(
            delayed(run_correlation)(goal_counts, tt) for tt in range(tscale.shape[0]))
        corr_goal = np.array([result[0] for result in results])
        sem_goal = np.array([result[1] for result in results])
        
        # Distractor object entries
        results = Parallel(n_jobs=-1)(
            delayed(run_correlation)(goal_counts, tt) for tt in range(tscale.shape[0]))        
        corr_dis = np.array([result[0] for result in results])
        sem_dis = np.array([result[1] for result in results])
        
        # Sound onset
        results = Parallel(n_jobs=-1)(
            delayed(run_correlation)(sound_counts, tt) for tt in range(tscale.shape[0]))        
        corr_sound = np.array([result[0] for result in results])
        sem_sound = np.array([result[1] for result in results])
        
        # Baseline subtract        
        corr_goal_bl = corr_goal - np.mean(corr_goal[tscale < 0])
        corr_dis_bl = corr_dis - np.mean(corr_dis[tscale < 0])
        corr_sound_bl = corr_sound - np.mean(corr_sound[tscale < 0])

        # Add to dataframe
        corr_df = pd.concat((corr_df, pd.DataFrame(data={
            'r_goal': corr_goal, 'r_goal_baseline': corr_goal_bl,
            'r_sem_goal': sem_goal, 'r_sem_distractor': sem_dis,
            'r_distractor': corr_dis, 'r_distractor_baseline': corr_dis_bl,
            'r_sound': corr_sound, 'r_sound_baseline': corr_sound_bl, 'r_sem_sound': sem_sound,
            'time': tscale, 'region_1': region_1, 'region_2': region_2,
            'region_pair': f'{region_1}-{region_2}',
            'subject': SUBJECT, 'date': DATE})), ignore_index=True)
        
# Save to disk
corr_df.to_csv(join(path_dict['save_path'], f'region_corr_{int(BIN_SIZE*1000)}ms-bins.csv'))