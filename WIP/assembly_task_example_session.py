# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:09:35 2025

By Guido Meijer & Gemini 2.5 Pro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pickle
from scipy.stats import zscore
from pathlib import Path
from msvr_functions import (paths, load_multiple_probes, calculate_peths, load_trials, figure_style,
                            peri_event_trace, load_objects)
colors, dpi = figure_style()


def detect_assemblies_from_task(task_spike_matrix):
    """
    Detects neural assemblies from a spike matrix of a specific period (e.g., a task).

    This function performs PCA on the z-scored spike matrix to find assembly patterns.
    It returns the significant principal components (the assemblies) and the
    statistics (mean and std) used for z-scoring, which are essential for
    projecting activity onto new data.

    Args:
        task_spike_matrix (np.ndarray): A 2D array of spike counts (neurons x timebins).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The significant eigenvectors (assemblies), shape (n_neurons, n_assemblies).
            - np.ndarray: The mean spike rate per neuron from the task period.
            - np.ndarray: The standard deviation of spike rates per neuron from the task period.
            - int: The number of detected assemblies.
    """
    # Ensure input is a 2D array
    task_spike_matrix = np.squeeze(task_spike_matrix)

    # Filter out neurons that have no spikes during the task period
    active_neuron_mask = np.sum(task_spike_matrix, axis=1) > 0
    if not np.any(active_neuron_mask):
        print("Warning: No active neurons in the provided task spike matrix.")
        return None, None, None, 0

    # Important: Keep track of which neurons were used to define the assemblies
    active_spike_matrix = task_spike_matrix[active_neuron_mask, :]
    n_neurons, n_timebins = active_spike_matrix.shape

    # 1. Z-score the spike matrix and get the parameters for later use
    # We need to calculate mean and std deviation along the timebins (axis=1)
    task_mean = np.mean(active_spike_matrix, axis=1, keepdims=True)
    task_std = np.std(active_spike_matrix, axis=1, keepdims=True)

    # Avoid division by zero for neurons with no variance in their firing rate
    task_std[task_std == 0] = 1
    spikemat_zscore = (active_spike_matrix - task_mean) / task_std

    # 2. Make correlation matrix
    corr_mat = np.corrcoef(spikemat_zscore)

    # 3. Get the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(corr_mat)

    # 4. Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. Determine number of assemblies using Marchenko-Pastur law
    q = n_timebins / n_neurons
    upper_bound = (1 + np.sqrt(1 / q))**2 if q > 0 else np.inf
    n_assemblies = np.sum(eigenvalues > upper_bound)
    print(f'Detected {n_assemblies} assemblies from the task period.')

    if n_assemblies == 0:
        print("No significant assemblies detected.")
        return None, task_mean, task_std, 0

    # 6. Isolate the significant eigenvectors (these are the cell assemblies)
    assemblies = eigenvectors[:, :n_assemblies]

    return assemblies, task_mean, task_std, n_assemblies, active_neuron_mask


def calculate_assembly_activation(
    new_spike_matrix, assemblies, task_mean, task_std, active_neuron_mask
):
    """
    Calculates the activation of pre-defined assemblies in a new spike matrix.

    Args:
        new_spike_matrix (np.ndarray): Spike matrix from a new period (e.g., rest).
                                       Shape should be (total_neurons, n_timebins).
        assemblies (np.ndarray): The assembly patterns (eigenvectors) from the task.
                                 Shape (n_active_neurons, n_assemblies).
        task_mean (np.ndarray): Mean spike rate per neuron from the original task period.
        task_std (np.ndarray): Std dev of spike rates from the original task period.
        active_neuron_mask (np.ndarray): Boolean mask indicating which neurons were used
                                         to define the assemblies.

    Returns:
        np.ndarray: A 2D array of assembly activations (n_assemblies x n_timebins).
    """
    if assemblies is None or assemblies.shape[1] == 0:
        print("No assemblies provided to calculate activation.")
        return np.array([])

    # Squeeze the matrix to ensure it's 2D
    new_spike_matrix = np.squeeze(new_spike_matrix)

    # IMPORTANT: Select only the neurons that were active and used to define the assemblies
    new_spike_matrix_active = new_spike_matrix[active_neuron_mask, :]

    # 1. Z-score the new spike matrix using the MEAN and STD from the TASK period
    new_spikemat_zscore = (new_spike_matrix_active - task_mean) / task_std

    # 2. Calculate assembly activation strength using the same formula
    # Term 1: Projection of activity onto each principal component, squared.
    projections = assemblies.T @ new_spikemat_zscore
    activation_main_term = projections**2

    # Term 2: Correction due to zeroing the diagonal of the outer product.
    correction_term = (assemblies**2).T @ (new_spikemat_zscore**2)

    # Final assembly activation is the difference between the two terms
    assembly_activation_matrix = activation_main_term - correction_term

    return assembly_activation_matrix


# %% Main script
# Settings
subject = '459601'
date = '20240411'
BIN_SIZE = 0.05
SMOOTHING = 0
T_BEFORE = 1
T_AFTER = 2

# Get paths
path_dict = paths()
session_path = Path(path_dict['local_data_path']) / 'Subjects' / subject / date

# Load in data
spikes, clusters, channels = load_multiple_probes(session_path)
trials = load_trials(subject, date)
obj_df = load_objects(subject, date)

# Generate pseudo reward times for unrewarded objects
reward_delays = (obj_df.loc[obj_df['rewarded'] == 1, 'reward_times']
                 - obj_df.loc[obj_df['rewarded'] == 1, 'times']).values
obj_df.loc[obj_df['goal'] == 0, 'rewarded'] = 0
obj_df.loc[obj_df['rewarded'] == 0, 'reward_times'] = (
    obj_df.loc[obj_df['rewarded'] == 0, 'times']
    + np.random.choice(reward_delays, size=np.sum(obj_df['rewarded'] == 0)))

# Loop over probes
for p, probe in enumerate(spikes.keys()):

    # Loop over regions
    for r, region in enumerate(np.unique(clusters[probe]['region'])):
        if region == 'root':
            continue
        print(f'Region {region}')

        # Get neurons in this region
        region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]

        # Create binned spike matrix of entire task period
        peths_task, binned_spikes_task = calculate_peths(
            spikes[probe]['times'], spikes[probe]['clusters'],
            region_neurons, [trials['enterEnvTime'].values[0]],
            pre_time=0, post_time=trials['enterEnvTime'].values[-1],
            bin_size=BIN_SIZE, smoothing=SMOOTHING)
        binned_spikes_task = np.squeeze(binned_spikes_task)  # (neurons x timebins)
        n_neurons, n_timebins = binned_spikes_task.shape
        time_ax = peths_task['tscale'] + trials['enterEnvTime'].values[0]

        # Detect assemblies
        assemblies, task_mean, task_std, n_assemblies, act_mask = detect_assemblies_from_task(binned_spikes_task)

        # Get assembly activation strenght during the task itself
        assembly_activation_task = calculate_assembly_activation(
            binned_spikes_task, assemblies, task_mean, task_std, act_mask)

        # Now project the detected assemblies on the rest period
        peths_rest, binned_spikes_rest = calculate_peths(
            spikes[probe]['times'], spikes[probe]['clusters'],
            region_neurons, [trials['exitEnvTime'].values[-1] + 60],
            pre_time=0, post_time=spikes[probe]['times'][-1] - 1,
            bin_size=BIN_SIZE, smoothing=SMOOTHING)
        binned_spikes_rest = np.squeeze(binned_spikes_rest)  # (neurons x timebins)
        assembly_activation_rest = calculate_assembly_activation(
            binned_spikes_rest, assemblies, task_mean, task_std, act_mask)

        # Plot assembly activity during task
        for ia in range(n_assemblies):
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75*3, 2.5), dpi=dpi, sharey=True)
            peri_event_trace(assembly_activation_task[ia, :], time_ax,
                             obj_df.loc[obj_df['object'] == 1, 'times'].values,
                             1 - obj_df.loc[obj_df['object'] == 1, 'goal'].values, ax=ax1,
                             t_before=T_BEFORE, t_after=T_AFTER, event_labels=['Goal', 'No goal'],
                             color_palette=[colors['goal'], colors['no-goal']])
            ax1.set(ylabel='Assembly activity', xlabel='Time from reward (s)', title='Object 1')

            peri_event_trace(assembly_activation_task[ia, :], time_ax,
                             obj_df.loc[obj_df['object'] == 2, 'times'].values,
                             1 - obj_df.loc[obj_df['object'] == 2, 'goal'].values, ax=ax2,
                             t_before=T_BEFORE, t_after=T_AFTER, event_labels=['Goal', 'No goal'],
                             color_palette=[colors['goal'], colors['no-goal']])
            ax2.set(xlabel='Time from reward (s)', title='Object 2')

            peri_event_trace(assembly_activation_task[ia, :], time_ax,
                             obj_df.loc[obj_df['object'] == 3, 'times'].values,
                             obj_df.loc[obj_df['object'] == 3, 'sound'].values, ax=ax3,
                             t_before=T_BEFORE, t_after=T_AFTER, event_labels=['Sound 1', 'Sound 2'],
                             color_palette=[colors['sound1'], colors['sound2']])
            ax3.set(xlabel='Time from reward (s)', title='Object 3')

            plt.suptitle(f'{region}')
            sns.despine(trim=True)
            plt.tight_layout()
            plt.show()
            #plt.savefig(path_dict['google_drive_fig_path'] / 'Assemblies'
            #            / f'{region}_{subject}_{date}_{ia}.jpg', dpi=600)
            #plt.close(f)




