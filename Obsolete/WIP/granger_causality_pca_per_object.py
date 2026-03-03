# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from itertools import combinations
from statsmodels.tsa.api import VAR
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from msvr_functions import (paths, load_multiple_probes, load_subjects, load_objects, figure_style,
                            calculate_peths)

# Settings
T_BEFORE = 1  # s
T_AFTER = 1
BIN_SIZE = 0.02
SMOOTHING = 0.03
MIN_NEURONS = 10
MIN_TRIALS = 30
N_CORES = -1
MAX_LAG = 0.5  # s
N_SHUFFLES = 500
N_COMPONENTS = 3

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
colors, dpi = figure_style()

def perform_pca_across_trials(binned_trials_3d, n_components=3):
    """
    Performs PCA across trials, fitting one model to all data.

    Args:
        binned_trials_3d (np.ndarray): A (trials x neurons x time_bins) matrix.
        n_components (int): The number of principal components to extract.

    Returns:
        np.ndarray: A (trials x time_bins x components) array of latent trajectories.
    """
    if binned_trials_3d is None or binned_trials_3d.size == 0:
        print("Cannot perform PCA on empty data.")
        return None

    n_trials, n_neurons, n_timebins = binned_trials_3d.shape
    
    # Reshape the data for PCA fitting.
    # We want a 2D matrix where each row is a time point and each column is a neuron.
    # The shape should be (total_time_points, n_neurons).
    # 1. Transpose from (trials, neurons, timebins) to (trials, timebins, neurons)
    # 2. Reshape to combine trials and timebins into one dimension.
    full_data_matrix = binned_trials_3d.transpose(0, 2, 1).reshape(-1, n_neurons)
    
    print(f"Fitting PCA with {n_components} components across {n_trials} trials.")
    pca = PCA(n_components=n_components)
    
    # Fit the model on the entire dataset and transform it.
    trajectories_flat = pca.fit_transform(full_data_matrix)
    print(f"PCA explained variance: {100 * pca.explained_variance_ratio_.sum():.2f}%")
    
    # Reshape the flat trajectories back into a 3D array (trials, timebins, components)
    trajectories_3d = trajectories_flat.reshape(n_trials, n_timebins, n_components)
        
    return trajectories_3d

def granger_causality(trial_data):
                
    # Region 1 to region 2
    model = VAR(trial_data)
    res = model.fit(maxlags=int(MAX_LAG / BIN_SIZE))
    f_test = res.test_causality('region1', 'region2', kind='f')
    f_12 = f_test.test_statistic
    
    # Region 2 to region 1
    f_test = res.test_causality('region2', 'region1', kind='f')
    f_21 = f_test.test_statistic

    return f_12, f_21

def granger_shuffled(i_shuf, prob_shuf, trial, this_obj, region1, region2):
    trial_data = pd.DataFrame({'region1': prob_shuf[this_obj][region1][trial, :, i_shuf],
                               'region2': prob_shuf[this_obj][region2][trial, :, i_shuf]})  
    # Region 1 to region 2
    model = VAR(trial_data)
    res = model.fit(maxlags=int(MAX_LAG / BIN_SIZE))
    f_test = res.test_causality('region1', 'region2', kind='f')
    f_12 = f_test.test_statistic
    
    # Region 2 to region 1
    f_test = res.test_causality('region2', 'region1', kind='f')
    f_21 = f_test.test_statistic
    
    return f_12, f_21    

# %% Main script
# Loop over recordings
granger_df = pd.DataFrame()
for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'\n{subject} {date} ({i} of {rec.shape[0]})')

    # Load in data
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_multiple_probes(session_path, min_fr=0.5)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)
    
    if trials.shape[0] < MIN_TRIALS:
        continue
    
    # Loop over regions
    # Get list of all regions and which probe they were recorded on
    regions, region_probes = [], []
    for p, probe in enumerate(spikes.keys()):
        regions.append(np.unique(clusters[probe]['region']))
        region_probes.append([probe] * np.unique(clusters[probe]['region']).shape[0])
    regions = np.concatenate(regions)
    region_probes = np.concatenate(region_probes)
    
    traj, traj['object1'], traj['object2'], traj['object3'] = dict(), dict(), dict(), dict()
    for r, (region, probe) in enumerate(zip(regions, region_probes)):
        if region == 'root':
            continue
        print(f'\nStarting {region}\n')
        
        # Get region neurons
        region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
        if region_neurons.shape[0] < MIN_NEURONS:
            continue
        
        # Loop over objects
        for this_obj in [1, 2, 3]:
            
            # Get binned spikes
            _, spike_counts = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], region_neurons, 
                all_obj_df.loc[all_obj_df['object'] == this_obj, 'times'].values,
                T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING )
            
            # Fit PCA
            traj[f'object{this_obj}'] = perform_pca_across_trials(spike_counts)  # (trials x timebins x PCs)
            
        # Plot this region
        f, axs = plt.subplots(1, 3, figsize=(6, 2), dpi=dpi)
        for isp, this_obj in enumerate([1, 2, 3]):
            for it in range(traj[f'object{this_obj}'].shape[0]):
                axs[isp].plot(traj[f'object{this_obj}'][it, :, 0], traj[f'object{this_obj}'][it, :, 1],
                              '-', lw=0.5, c='grey', alpha=0.25)
            average_trajectory = np.mean(traj[f'object{this_obj}'], axis=0) 
            axs[isp].plot(average_trajectory[:, 0], average_trajectory[:, 1], color='orange', lw=1.5)
            axs[isp].set(title=f'Object {this_obj}')
        f.suptitle='f{region}'
        plt.tight_layout()
        
        plt.savefig(join(path_dict['fig_path'], 'PCA', f'{region}_{subject}_{date}.jpg'))
        plt.close(f)
        
    # Do Granger causality for all region pairs
    print('Run Granger causality..')
    for this_obj in ['object1', 'object2', 'object3']:
        for region1, region2 in combinations(traj[this_obj].keys(), 2):
            
            # Do Granger causality per trial
            n_trials = traj[this_obj][region1].shape[2]
            reg1_reg2, reg2_reg1 = np.empty(n_trials), np.empty(n_trials)
            reg1_reg2_p, reg2_reg1_p = np.empty(n_trials), np.empty(n_trials) 
            for trial in range(n_trials):
                
                # Granger causality
                asd
                trial_data = pd.DataFrame({'region1': traj[this_obj][region1][trial],
                                           'region2': traj[this_obj][region2][trial]})  
                reg1_reg2[trial], reg2_reg1[trial] = granger_causality(trial_data)
            
                # Shuffles
                results = Parallel(n_jobs=N_CORES)(delayed(granger_shuffled)(
                    i_shuf, prob_shuf, trial, this_obj, region1, region2) for i_shuf in range(N_SHUFFLES))
                reg1_reg2_shuf = np.array([i[0] for i in results])
                reg2_reg1_shuf = np.array([i[1] for i in results])
  
                # Get p-value
                z = (reg1_reg2[trial] - np.mean(reg1_reg2_shuf)) / np.std(reg1_reg2_shuf)
                reg1_reg2_p[trial] = 2 * (1 - stats.norm.cdf(abs(z)))
                z = (reg2_reg1[trial] - np.mean(reg2_reg1_shuf)) / np.std(reg2_reg1_shuf)
                reg2_reg1_p[trial] = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # Get p-value over trials by doing a binomial test
            p_reg1_reg2 = stats.binomtest(np.sum(reg1_reg2_p < 0.05), n_trials, 0.05).pvalue
            p_reg2_reg1 = stats.binomtest(np.sum(reg2_reg1_p < 0.05), n_trials, 0.05).pvalue
            
            # Add to dataframe
            granger_df = pd.concat((granger_df, pd.DataFrame(data={
                'region_pair': [f'{region1} → {region2}', f'{region2} → {region1}'],
                'region1': [region1, region2], 'region2': [region2, region1],
                'object': this_obj,
                'p_value': [p_reg1_reg2, p_reg2_reg1],
                'f_stat': [np.median(reg1_reg2), np.median(reg2_reg1)],
                'subject': subject, 'date': date})))
        
    # Save to disk
    granger_df.to_csv(join(path_dict['save_path'], 'granger_causality_objects.csv'), index=False)
    