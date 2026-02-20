# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import FastICA
from msvr_functions import (paths, load_neural_data, load_objects, combine_regions,
                            calculate_peths, figure_style)
colors, dpi = figure_style()

# Settings
BIN_SIZE = 0.05
MIN_NEURONS = 5
MIN_RIPPLES = 0
SMOOTHING_SIGMA = 2
MP_THRESHOLD_SCALE = 1.2  # Scale factor for Marchenko-Pastur threshold (higher -> fewer assemblies)

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)

# %% MAIN

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

        # Get region neurons
        region_neurons = clusters['cluster_id'][clusters['region'] == region]
        region_spikes = spikes['times'][np.isin(spikes['clusters'], region_neurons)]
        region_clusters = spikes['clusters'][np.isin(spikes['clusters'], region_neurons)]
        if np.unique(region_clusters).shape[0] < MIN_NEURONS:
            continue

        # Create binned spike matrix of entire task period
        peths_task, binned_spikes = calculate_peths(
            spikes['times'], spikes['clusters'],
            region_neurons, [0],
            pre_time=0, post_time=spikes['times'][-1],
            bin_size=BIN_SIZE, smoothing=0)
        binned_spikes = np.squeeze(binned_spikes)  # (neurons x timebins)
        n_timebins = binned_spikes.shape[1]
        binned_time = peths_task['tscale']

        # Remove silent neurons
        active_idx = np.std(binned_spikes, axis=1) > 0
        binned_spikes = binned_spikes[active_idx]
        n_neurons = binned_spikes.shape[0]

        if n_neurons < MIN_NEURONS:
            continue

        # Z-score spikes
        z_spikes = (binned_spikes - np.mean(binned_spikes, axis=1, keepdims=True)) / np.std(binned_spikes, axis=1, keepdims=True)

        # Correlation matrix
        corr_mat = np.corrcoef(z_spikes)

        # Eigen decomposition
        evals, evecs = np.linalg.eigh(corr_mat)
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Marchenko-Pastur threshold
        q = n_neurons / n_timebins
        lambda_max = ((1 + np.sqrt(q)) ** 2) * MP_THRESHOLD_SCALE

        # Number of assemblies
        n_assemblies = np.sum(evals > lambda_max)
        print(f'Detected {n_assemblies} assemblies')
        if n_assemblies == 0:
            continue

        # PCA projection
        pcs = evecs[:, :n_assemblies]
        projected_data = pcs.T @ z_spikes

        # ICA
        ica = FastICA(n_components=n_assemblies, random_state=0)
        ica.fit(projected_data.T)

        # Assembly patterns (neurons x assemblies)
        assembly_patterns = pcs @ ica.mixing_

        # Correct sign (make max absolute weight positive)
        for k in range(n_assemblies):
            if np.abs(assembly_patterns[:, k].min()) > np.abs(assembly_patterns[:, k].max()):
                assembly_patterns[:, k] *= -1

        # Assembly activations (assemblies x timebins)
        assembly_activations = assembly_patterns.T @ z_spikes
        assembly_activations = gaussian_filter(assembly_activations, SMOOTHING_SIGMA)

        # Save assemblies to disk
        np.save(path_dict['google_drive_data_path'] / 'Assemblies' / f'{subject}_{date}_{region}.amplitudes.npy',
                assembly_activations)
        np.save(path_dict['google_drive_data_path'] / 'Assemblies' / f'{subject}_{date}_{region}.times.npy',
                binned_time)