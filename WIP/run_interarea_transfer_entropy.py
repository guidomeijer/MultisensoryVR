# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from pyinform.transferentropy import transfer_entropy
from joblib import Parallel, delayed
from scipy.stats import pearsonr, sem
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
T_BEFORE = 2
T_AFTER = 2
BIN_SIZE = 0.15
SMOOTHING = 0
MIN_FR = 0.5  # minimum firing rate over whole recording in spks/s
MIN_NEURONS = 10  # per region
K = 0.5  # history length in seconds

# Initialize
path_dict = paths()
subjects = load_subjects()

# Load in data
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv'))
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')

trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
all_obj_df = load_objects(SUBJECT, DATE)
rew_obj2_df = all_obj_df[all_obj_df['object'] == 2]

# %% Fuction for parallization

def run_te(spike_counts, tt):
    pairwise_te = []
    for n1 in range(spike_counts[region_1].shape[1]):  # Neurons in region 1
        for n2 in range(spike_counts[region_2].shape[1]):  # Neurons in region 2
            te = transfer_entropy(spike_counts[region_1][:, n1, tt],
                                  spike_counts[region_2][:, n2, tt],
                                  int(K / BIN_SIZE))
            pairwise_te.append(te)
    te_mean = np.nanmean(pairwise_te)
    te_sem = sem(pairwise_te, nan_policy='omit')
  
    return te_mean, te_sem


# %%


for i, (subject, date) in enumerate(zip(np.unique(rec['subject']), np.unique(rec['date']))):

    # Load in neural data for all probes
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_multiple_probes(session_path, min_fr=MIN_FR)

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
    for r1, region_1 in enumerate(these_regions):
        for r2, region_2 in enumerate(these_regions):
            if region_1 == region_2:
                continue
            print(f'{region_1} - {region_2}')
            
            # Goal object entries
            results = Parallel(n_jobs=-1)(
                delayed(run_te)(goal_counts, tt) for tt in range(tscale.shape[0]))
            te_goal = np.array([result[0] for result in results])
            sem_goal = np.array([result[1] for result in results])
            
            # Distractor object entries
            results = Parallel(n_jobs=-1)(
                delayed(run_te)(distractor_counts, tt) for tt in range(tscale.shape[0]))        
            te_dis = np.array([result[0] for result in results])
            sem_dis = np.array([result[1] for result in results])
            
            # Sound onset
            results = Parallel(n_jobs=-1)(
                delayed(run_te)(sound_counts, tt) for tt in range(tscale.shape[0]))        
            te_sound = np.array([result[0] for result in results])
            sem_sound = np.array([result[1] for result in results])
            
            # Baseline subtract        
            te_goal_bl = te_goal - np.mean(te_goal[tscale < -1])
            te_dis_bl = te_dis - np.mean(te_dis[tscale < -1])
            te_sound_bl = te_sound - np.mean(te_sound[tscale < -1])
    
            # Add to dataframe
            mi_df = pd.concat((mi_df, pd.DataFrame(data={
                'te_goal': te_goal, 'te_goal_baseline': te_goal_bl,
                'te_sem_goal': sem_goal, 'te_sem_distractor': sem_dis,
                'te_distractor': te_dis, 'te_distractor_baseline': te_dis_bl,
                'te_sound': te_sound, 'te_sound_baseline': te_sound_bl, 'te_sem_sound': sem_sound,
                'time': tscale, 'region_1': region_1, 'region_2': region_2,
                'region_pair': f'{region_1}-{region_2}',
                'subject': SUBJECT, 'date': DATE})), ignore_index=True)
            
    # Save to disk
    mi_df.to_csv(join(path_dict['save_path'], f'region_te_{int(BIN_SIZE*1000)}ms-bins.csv'))