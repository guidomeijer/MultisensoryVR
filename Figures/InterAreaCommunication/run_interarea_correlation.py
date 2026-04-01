# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, sem
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
T_BEFORE = 2
T_AFTER = 2
BIN_SIZE = 0.1
SMOOTHING = 0.1
SUBTRACT_MEAN = True
MIN_NEURONS = 5  # per region
N_CPUS = 18

# Initialize
path_dict = paths()
subjects = load_subjects()

# Load in data
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(subset=['subject', 'date'])
neurons_df = pd.read_csv(path_dict['save_path'] / 'significant_neurons.csv')
neurons_df['subject'] = neurons_df['subject'].astype(str)
neurons_df['date'] = neurons_df['date'].astype(str)

# Select neurons to include
neurons_df['include'] = (neurons_df['p_context_obj2'] < 0.05) | (neurons_df['p_obj_onset'] < 0.05) | (neurons_df['p_reward'] < 0.05)

# %% Fuction for parallization

def run_correlation(spike_counts, reg1, reg2, tt):
    # Extract data for this time bin: (trials x neurons)
    x = spike_counts[reg1][:, :, tt]
    y = spike_counts[reg2][:, :, tt]

    # Calculate covariance matrix using matrix multiplication.
    # Since the mean is already subtracted over trials in the main loop,
    # the dot product divided by the number of trials gives the covariance.
    corr_matrix = (x.T @ y) / x.shape[0]

    pairwise_corr = corr_matrix.flatten()
    r_mean = np.nanmean(pairwise_corr)
    r_sem = sem(pairwise_corr, nan_policy='omit')

    return r_mean, r_sem


# %%
corr_df = pd.DataFrame()

for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'Recording {i} of {len(rec)}: \n{subject} {date}')

    # Load in neural data for all probes
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
    spikes, clusters, channels = load_multiple_probes(session_path)

    # Load in object entry times
    all_obj_df = load_objects(subject, date)
    trials = pd.read_csv(path_dict['local_data_path'] / 'Subjects' / subject / date / 'trials.csv')

    # Get binned spike counts per region
    obj1_rew, obj1_no_rew, obj2_rew, obj2_no_rew = dict(), dict(), dict(), dict()
    for k, probe in enumerate(spikes.keys()):
        for j, region in enumerate(np.unique(clusters[probe]['region'])):
            if region == 'root':
                continue
            
            # Select neurons
            region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
            sig_neurons = neurons_df.loc[(neurons_df['include']
                                          & (neurons_df['subject'] == subject)
                                          & (neurons_df['date'] == date)
                                          & (neurons_df['probe'] == probe)), 'neuron_id'].values
            use_neurons = region_neurons[np.isin(region_neurons, sig_neurons)]
            if use_neurons.shape[0] < MIN_NEURONS:
                continue
        
            # Get spike counts (trials x neurons x time)
            _, obj1_rew[region] = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], use_neurons,
                all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['object'] == 1), 'times'],
                T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING, return_fr=False)
            _, obj1_no_rew[region] = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], use_neurons,
                all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] == 1), 'times'],
                T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING, return_fr=False)
            _, obj2_rew[region] = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], use_neurons,
                all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['object'] == 2), 'times'],
                T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING, return_fr=False)
            peths, obj2_no_rew[region] = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], use_neurons,
                all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] == 2), 'times'],
                T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING, return_fr=False)
        
            # Get time scale
            tscale = peths['tscale']

            # Subtract mean
            if SUBTRACT_MEAN:
                obj1_rew[region] = obj1_rew[region] - np.mean(obj1_rew[region], axis=0)
                obj1_no_rew[region] = obj1_no_rew[region] - np.mean(obj1_no_rew[region], axis=0)
                obj2_rew[region] = obj2_rew[region] - np.mean(obj2_rew[region], axis=0)
                obj2_no_rew[region] = obj2_no_rew[region] - np.mean(obj2_no_rew[region], axis=0)

    # Get pairwise neural correlations between all neuron pairs in both regions
    these_regions = list(obj1_rew.keys())
    for r1, region_1 in enumerate(these_regions[:-1]):
        for r2, region_2 in enumerate(these_regions[r1+1:]):
            print(f'{region_1} - {region_2}')
            
            # Object 1
            results = Parallel(n_jobs=N_CPUS)(
                delayed(run_correlation)(obj1_rew, region_1, region_2, tt) for tt in range(tscale.shape[0]))
            corr_obj1_rew = np.array([result[0] for result in results])
            sem_obj1_rew = np.array([result[1] for result in results])
            results = Parallel(n_jobs=N_CPUS)(
                delayed(run_correlation)(obj1_no_rew, region_1, region_2, tt) for tt in range(tscale.shape[0]))
            corr_obj1_no_rew = np.array([result[0] for result in results])
            sem_obj1_no_rew = np.array([result[1] for result in results])

            # Object 2
            results = Parallel(n_jobs=N_CPUS)(
                delayed(run_correlation)(obj2_rew, region_1, region_2, tt) for tt in range(tscale.shape[0]))
            corr_obj2_rew = np.array([result[0] for result in results])
            sem_obj2_rew = np.array([result[1] for result in results])
            results = Parallel(n_jobs=N_CPUS)(
                delayed(run_correlation)(obj2_no_rew, region_1, region_2, tt) for tt in range(tscale.shape[0]))
            corr_obj2_no_rew = np.array([result[0] for result in results])
            sem_obj2_no_rew = np.array([result[1] for result in results])
            
            # Baseline subtract        
            corr_obj1_rew_bl = corr_obj1_rew - np.mean(corr_obj1_rew[tscale < -1])
            corr_obj1_no_rew_bl = corr_obj1_no_rew - np.mean(corr_obj1_no_rew[tscale < -1])
            corr_obj2_rew_bl = corr_obj2_rew - np.mean(corr_obj2_rew[tscale < -1])
            corr_obj2_no_rew_bl = corr_obj2_no_rew - np.mean(corr_obj2_no_rew[tscale < -1])
    
            # Add to dataframe
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'r_obj1_rew': corr_obj1_rew, 'r_obj1_rew_baseline': corr_obj1_rew_bl,
                'r_obj1_no_rew': corr_obj1_no_rew, 'r_obj1_no_rew_baseline': corr_obj1_no_rew_bl,
                'r_obj2_rew': corr_obj2_rew, 'r_obj2_rew_baseline': corr_obj2_rew_bl,
                'r_obj2_no_rew': corr_obj2_no_rew, 'r_obj2_no_rew_baseline': corr_obj2_no_rew_bl,
                'time': tscale, 'region_1': region_1, 'region_2': region_2,
                'region_pair': f'{region_1}-{region_2}',
                'subject': subject, 'date': date})), ignore_index=True)
            
    # Save to disk
    corr_df.to_csv(path_dict['google_drive_data_path'] / f'region_corr_{int(BIN_SIZE*1000)}ms-bins.csv')