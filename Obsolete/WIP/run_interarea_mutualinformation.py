# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from pyinform.mutualinfo import mutual_info
from joblib import Parallel, delayed
from scipy.stats import pearsonr, sem
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
T_BEFORE = 2
T_AFTER = 2
BIN_SIZE = 0.15
SMOOTHING = 0
SUBTRACT_MEAN = False
MIN_NEURONS = 10  # per region
MIN_FR = 0.5

# Initialize
path_dict = paths()
subjects = load_subjects()

# Load in data
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv'))
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))

# %% Fuction for parallization

def run_mi(spike_counts, tt):
    pairwise_mi = []
    for n1 in range(spike_counts[region_1].shape[1]):  # Neurons in region 1
        for n2 in range(spike_counts[region_2].shape[1]):  # Neurons in region 2
            mi = mutual_info(spike_counts[region_1][:, n1, tt],
                             spike_counts[region_2][:, n2, tt])
            pairwise_mi.append(mi)
    mi_mean = np.nanmean(pairwise_mi)
    mi_sem = sem(pairwise_mi, nan_policy='omit')
  
    return mi_mean, mi_sem


# %%


for i, (subject, date) in enumerate(zip(np.unique(rec['subject']), np.unique(rec['date']))):

    # Load in data for this session
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', subject, date, 'trials.csv'))
    spikes, clusters, channels = load_multiple_probes(session_path, min_fr=MIN_FR)
    all_obj_df = load_objects(subject, date)
    rew_obj2_df = all_obj_df[all_obj_df['object'] == 2]

    # Get binned spike counts per region
    goal_counts, distractor_counts, sound_counts = dict(), dict(), dict()
    mi_df = pd.DataFrame()
    for k, probe in enumerate(spikes.keys()):
        for j, region in enumerate(np.unique(clusters[probe]['region'])):
            if region == 'root':
                continue
            
            # Select neurons
            region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
            #sig_neurons = neurons_df.loc[neurons_df['sig_goal'] | neurons_df['sig_obj_onset'], 'neuron_id'].values
            sig_neurons = neurons_df.loc[(neurons_df['sig_goal']
                                          & (neurons_df['subject'] == subject)
                                          & (neurons_df['date'] == date)
                                          & (neurons_df['probe'] == probe)), 'neuron_id'].values
            #sig_neurons = neurons_df.loc[neurons_df['sig_goal'], 'neuron_id'].values
            #use_neurons = region_neurons[np.isin(region_neurons, sig_neurons)]
            use_neurons = region_neurons
            if use_neurons.shape[0] < MIN_NEURONS:
                continue
        
            # Get spike counts
            peths, goal_counts[region] = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], use_neurons,
                rew_obj2_df.loc[rew_obj2_df['goal'] == 1, 'times'], T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING,
                return_fr=False)
            
            peths, distractor_counts[region] = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], use_neurons,
                rew_obj2_df.loc[rew_obj2_df['goal'] == 0, 'times'], T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING,
                return_fr=False)
            
            peths, sound_counts[region] = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], use_neurons,
                trials['soundOnset'].values, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING,
                return_fr=False)
        
            # Get time scale
            tscale = peths['tscale']
    
    # Get pairwise mutual information between all neuron pairs in both regions
    these_regions = list(goal_counts.keys())
    for r1, region_1 in enumerate(these_regions[:-1]):
        for r2, region_2 in enumerate(these_regions[r1+1:]):
            print(f'{region_1} - {region_2}')
            
            # Goal object entries
            results = Parallel(n_jobs=-1)(
                delayed(run_mi)(goal_counts, tt) for tt in range(tscale.shape[0]))
            mi_goal = np.array([result[0] for result in results])
            sem_goal = np.array([result[1] for result in results])
            
            # Distractor object entries
            results = Parallel(n_jobs=-1)(
                delayed(run_mi)(distractor_counts, tt) for tt in range(tscale.shape[0]))        
            mi_dis = np.array([result[0] for result in results])
            sem_dis = np.array([result[1] for result in results])
            
            # Sound onset
            results = Parallel(n_jobs=-1)(
                delayed(run_mi)(sound_counts, tt) for tt in range(tscale.shape[0]))        
            mi_sound = np.array([result[0] for result in results])
            sem_sound = np.array([result[1] for result in results])
            
            # Baseline subtract        
            mi_goal_bl = mi_goal - np.mean(mi_goal[tscale < -1])
            mi_dis_bl = mi_dis - np.mean(mi_dis[tscale < -1])
            mi_sound_bl = mi_sound - np.mean(mi_sound[tscale < -1])
    
            # Add to dataframe
            mi_df = pd.concat((mi_df, pd.DataFrame(data={
                'mi_goal': mi_goal, 'mi_goal_baseline': mi_goal_bl,
                'mi_sem_goal': sem_goal, 'mi_sem_distractor': sem_dis,
                'mi_distractor': mi_dis, 'mi_distractor_baseline': mi_dis_bl,
                'mi_sound': mi_sound, 'mi_sound_baseline': mi_sound_bl, 'mi_sem_sound': sem_sound,
                'time': tscale, 'region_1': region_1, 'region_2': region_2,
                'region_pair': f'{region_1}-{region_2}',
                'subject': SUBJECT, 'date': DATE})), ignore_index=True)
            
    # Save to disk
    mi_df.to_csv(join(path_dict['save_path'], f'region_mi_{int(BIN_SIZE*1000)}ms-bins.csv'))