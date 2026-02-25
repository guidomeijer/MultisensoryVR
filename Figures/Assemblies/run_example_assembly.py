# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 24/02/2026
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import FastICA
from msvr_functions import (paths, load_neural_data, load_objects,
                            calculate_peths, figure_style)
colors, dpi = figure_style()

# Settings
SUBJECT = '462910'
DATE = '20240815'
PROBE = 'probe00'
REGION = 'CA1'
BIN_SIZE = 0.05
MIN_NEURONS = 5
SMOOTHING_SIGMA = 1
MP_THRESHOLD_SCALE = 1.5  # Scale factor for Marchenko-Pastur threshold (higher -> fewer assemblies)

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)

# %% MAIN

# Load in data
session_path = path_dict['local_data_path'] / 'Subjects' / f'{SUBJECT}' / f'{DATE}'
spikes, clusters, channels = load_neural_data(session_path, PROBE)
trials = pd.read_csv(path_dict['local_data_path'] / 'Subjects' / SUBJECT / DATE / 'trials.csv')
all_obj_df = load_objects(SUBJECT, DATE)

# Get region neurons
region_neurons = clusters['cluster_id'][clusters['region'] == REGION]
region_spikes = spikes['times'][np.isin(spikes['clusters'], region_neurons)]
region_clusters = spikes['clusters'][np.isin(spikes['clusters'], region_neurons)]

# Create binned spike matrix of entire task period
peths_task, binned_spikes = calculate_peths(
    spikes['times'], spikes['clusters'],
    region_neurons, [0],
    pre_time=0, post_time=spikes['times'][-1],
    bin_size=BIN_SIZE, smoothing=0)
binned_spikes = binned_spikes[0]  # (neurons x timebins)
n_timebins = binned_spikes.shape[1]
binned_time = peths_task['tscale']

# Remove silent neurons
active_idx = np.std(binned_spikes, axis=1) > 0
binned_spikes = binned_spikes[active_idx]
n_neurons = binned_spikes.shape[0]

# Z-score spikes
z_spikes = (binned_spikes - np.mean(binned_spikes, axis=1, keepdims=True)) / np.std(binned_spikes, axis=1, keepdims=True)

# Correlation matrix
corr_arr = np.corrcoef(z_spikes)

# Eigen decomposition
evals, evecs = np.linalg.eigh(corr_arr)
idx = evals.argsort()[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

# Marchenko-Pastur threshold
q = n_neurons / n_timebins
lambda_max = ((1 + np.sqrt(q)) ** 2) * MP_THRESHOLD_SCALE

# Number of assemblies
n_assemblies = np.sum(evals > lambda_max)
print(f'Detected {n_assemblies} assemblies')

# PCA projection
pcs = evecs[:, :n_assemblies]
projected_data = pcs.T @ z_spikes

# ICA
ica = FastICA(n_components=n_assemblies, random_state=0)
activations = ica.fit_transform(projected_data.T).T

# Assembly patterns (neurons x assemblies)
assembly_patterns = pcs @ ica.mixing_

# Correct sign (make max absolute weight positive)
for k in range(n_assemblies):
    if np.abs(assembly_patterns[:, k].min()) > np.abs(assembly_patterns[:, k].max()):
        assembly_patterns[:, k] *= -1
        activations[k, :] *= -1

# Assembly activations (assemblies x timebins)
assembly_activations = gaussian_filter(activations, [0, SMOOTHING_SIGMA])

# %% Plotting

f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax.plot(np.arange(1, evals.shape[0]+1), evals)
ax.plot([0, evals.shape[0]], [lambda_max, lambda_max], ls='--', color='red')
ax.set(xlim=[0, 40], ylabel='Eigenvalue', xlabel='Principal component',
       xticks=[1, 10, 20, 30, 40], yticks=[0, 2, 4, 6, 8])

sns.despine(trim=True)
plt.tight_layout()
plt.show()


# %%
# Get active cluster IDs
active_clusters = np.array(region_neurons)[active_idx]

for i in range(n_assemblies):
    fig, ax = plt.subplots(3, 1, figsize=(5, 8), dpi=dpi)

    # 1. Assembly weights
    weights = assembly_patterns[:, i]
    # Sort by weight
    sort_idx = np.argsort(weights)
    ax[0].stem(weights[sort_idx], basefmt=' ')
    ax[0].set_ylabel('Weight')
    ax[0].set_xlabel('Neuron (sorted)')
    ax[0].set_title(f'Assembly {i+1}')

    # Find a time window with high activation
    act = assembly_activations[i, :]
    peak_idx = np.argmax(act)
    center_time = binned_time[peak_idx]
    win = 2  # seconds

    # 2. Raster plot of high-weight neurons
    # Define high weight neurons (e.g., > 2 std)
    thresh = np.mean(weights) + 2 * np.std(weights)
    high_weight_indices = np.where(weights > thresh)[0]

    # Sort these by weight
    hw_weights = weights[high_weight_indices]
    sorted_hw_indices = high_weight_indices[np.argsort(hw_weights)]

    # Plot raster
    for y, idx in enumerate(sorted_hw_indices):
        cluster_id = active_clusters[idx]
        st = spikes['times'][spikes['clusters'] == cluster_id]
        st_win = st[(st >= center_time - win) & (st <= center_time + win)]
        ax[1].vlines(st_win, y, y + 1, color='k')

    ax[1].set_ylim(0, len(high_weight_indices))
    ax[1].set_ylabel('Neurons (high weight)')
    ax[1].set_xlim(center_time - win, center_time + win)
    ax[1].set_xticks([])  # Share x with bottom

    # 3. Activation
    t_mask = (binned_time >= center_time - win) & (binned_time <= center_time + win)
    ax[2].plot(binned_time[t_mask], act[t_mask])
    ax[2].set_ylabel('Activation')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_xlim(center_time - win, center_time + win)

    plt.tight_layout()
    plt.show()