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
import matplotlib.pyplot as plt
import neo
from elephant.gpfa import GPFA
import quantities as pq
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from msvr_functions import paths, load_multiple_probes, load_subjects, load_objects, figure_style

# Settings
T_BEFORE = 2  # s
T_AFTER = 1
BIN_SIZE = 0.1
MIN_NEURONS = 10
MIN_TRIALS = 30
N_CORES = -1
MAX_LAG = 0.5  # s
N_SHUFFLES = 500
N_COMPONENTS = 3

# Initialize
clf = LogisticRegression(solver='liblinear')
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
colors, dpi = figure_style()

def create_spiketrains_for_trial(spike_times, spike_ids, selected_neuron_ids, trial_start_time, trial_duration_s):
    """
    Creates a list of neo.SpikeTrain objects for a single trial from raw spike data,
    using only the neurons specified in selected_neuron_ids.
    
    Args:
        spike_times (np.ndarray): Array of all spike times.
        spike_ids (np.ndarray): Array of neuron IDs for each spike.
        selected_neuron_ids (list or np.ndarray): A list of neuron IDs to include.
        trial_start_time (float): The start time of the trial in seconds.
        trial_duration_s (float): The duration of the trial in seconds.

    Returns:
        list: A list of neo.SpikeTrain objects.
    """
    trial_spiketrains = []
    trial_end_time = trial_start_time + trial_duration_s
    
    # Iterate only through the selected neuron IDs
    for neuron_id in selected_neuron_ids:
        # 1. Filter all spikes to get just those from the current neuron
        neuron_mask = (spike_ids == neuron_id)
        neuron_spike_times = spike_times[neuron_mask]
        
        # 2. Filter the neuron's spikes to get just those in the current trial
        trial_mask = (neuron_spike_times >= trial_start_time) & (neuron_spike_times < trial_end_time)
        spikes_in_trial = neuron_spike_times[trial_mask]
        
        # 3. Make spike times relative to the start of the trial
        relative_spike_times = spikes_in_trial - trial_start_time
        
        # 4. Create the neo.SpikeTrain object with correct units and duration
        st = neo.SpikeTrain(relative_spike_times * pq.s, t_stop=trial_duration_s * pq.s)
        trial_spiketrains.append(st)
        
    return trial_spiketrains

def perform_gpfa(spiketrains, bin_size_s, n_components=3):
    """
    Performs GPFA directly on a list of SpikeTrain objects.
    """
    # Robustness check: Ensure there are enough active neurons to fit the model.
    active_neurons = [st for st in spiketrains if len(st) > 0]
    if len(active_neurons) < n_components:
        print(f"Skipping GPFA: Only {len(active_neurons)} active neurons found (need at least {n_components}).")
        # Determine number of bins from one of the spiketrains
        n_bins = int(spiketrains[0].t_stop / (bin_size_s * pq.s))
        return np.full((n_bins, n_components), np.nan)

    gpfa = GPFA(bin_size=bin_size_s * pq.s, x_dim=n_components)
    
    try:
        latent_variables = gpfa.fit_transform(active_neurons)
    except Exception as e:
        print(f"GPFA fitting failed: {e}")
        n_bins = int(spiketrains[0].t_stop / (bin_size_s * pq.s))
        return np.full((n_bins, n_components), np.nan)

    # Output is (n_components x n_bins), transpose to (n_bins x n_components)
    return latent_variables.T

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
        
            # Create list of spike train objects
            spiketrains_per_trial = []
            for t, trial_time in enumerate(all_obj_df.loc[all_obj_df['object'] == this_obj, 'times'].values):
                
                these_spiketrains = create_spiketrains_for_trial(
                    spikes[probe]['times'], spikes[probe]['clusters'], region_neurons,
                    trial_start_time=trial_time - T_BEFORE, trial_duration_s=T_BEFORE + T_AFTER)
                spiketrains_per_trial.append(these_spiketrains)
            
            # Fit GPFA
            trajectories = perform_gpfa(spiketrains_per_trial, BIN_SIZE, n_components=N_COMPONENTS)
            traj[f'object{this_obj}'] = np.dstack(trajectories)  # (components, timebins, trials)
            
        # Plot this region
        f, axs = plt.subplots(1, 3, figsize=(6, 2), dpi=dpi)
        for isp, this_obj in enumerate([1, 2, 3]):
            for it in range(traj[f'object{this_obj}'].shape[2]):
                axs[isp].plot(traj[f'object{this_obj}'][0, :, it], traj[f'object{this_obj}'][1, :, it],
                              '-', lw=0.5, c='grey', alpha=0.25)
            average_trajectory = np.mean(traj[f'object{this_obj}'], axis=2) 
            axs[isp].plot(average_trajectory[0, :], average_trajectory[1, :], color='orange', lw=1.5)
            axs[isp].set(title=f'Object {this_obj}')
        f.suptitle='f{region}'
        plt.tight_layout()
        plt.savefig(join(path_dict['fig_path'], 'GPFA', f'{region}_{subject}_{date}.jpg'))
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
    