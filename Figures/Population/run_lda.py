# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:15:00 2026

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy import stats
from itertools import combinations
import mne
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from msvr_functions import paths, load_subjects, figure_style, load_objects, add_significance
colors, dpi = figure_style()
mne.set_log_level('WARNING')

# Settings
MIN_NEURONS = 5
FIT_POS = [700, 750]  # mm
REGION_ORDER = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']
REGION_NAMES = ['Visual cortex', 'Auditory cortex', 'Temporal association cortex', 'Perirhinal cortex',
                'Lateral entorhinal cortex', 'Hippocampus (CA1)']

# Load in data
path_dict = paths()
subjects = load_subjects()    
with open(path_dict['google_drive_data_path'] / 'residuals_position_50mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Add neuron type to spike_dict
neuron_type = pd.read_csv(path_dict['save_path'] / 'waveform_metrics.csv',
                          dtype={'subject': str, 'date': str})

# Group recordings by date (simultaneous recordings)
unique_dates = np.unique(spike_dict['date'])

results_df = pd.DataFrame()
for date in unique_dates:

    # Find all recording indices for this date
    rec_indices = [idx for idx in range(len(spike_dict['date'])) if spike_dict['date'][idx] == date]
    this_subject = spike_dict['subject'][rec_indices[0]]
    
    # Get whether this is a FAR or NEAR session
    is_far = subjects.loc[subjects['SubjectID'] == this_subject, 'Far'].values[0]
    if is_far == 0:
        continue
    
    # Load in object data
    obj_df = load_objects(this_subject, str(date))

    # Get which context is the rewarded context for the first and second object
    obj1_goal = obj_df.loc[(obj_df['object'] == 1) & (obj_df['goal'] == 1), 'sound'].values[0]
    obj2_goal = obj_df.loc[(obj_df['object'] == 2) & (obj_df['goal'] == 1), 'sound'].values[0]

    # Find all unique regions recorded on this date across all probes
    session_regions = []
    for idx in rec_indices:
        session_regions.extend(np.unique(spike_dict['region'][idx]))
    session_regions = [r for r in np.unique(session_regions) if r != 'root']

    # Project neural activity onto context projection for each brain region
    region_latents = {}
    spatial_bins = spike_dict['position'][rec_indices[0]].astype(int)
    context_per_bin = spike_dict['context'][rec_indices[0]]
    unique_positions = np.unique(spatial_bins)
    n_bins = len(unique_positions)

    n_trials_A = spatial_bins[context_per_bin == obj1_goal].shape[0] // n_bins
    n_trials_B = spatial_bins[context_per_bin == 3 - obj1_goal].shape[0] // n_bins
    min_trials = np.min([n_trials_A, n_trials_B])
    if min_trials < 2:
        continue

    for region in session_regions:
        # Collect neurons for this region across probes
        region_spikes = []
        for idx in rec_indices:
            region_spikes.append(spike_dict['residuals'][idx][:, spike_dict['region'][idx] == region])
        if len(region_spikes) == 0:
            continue
        spike_counts = np.hstack(region_spikes)

        # Throw out silent neurons
        spike_counts = spike_counts[:, np.std(spike_counts, axis=0) > 0.5]

        # If not enough neurons, continue
        if spike_counts.shape[1] < MIN_NEURONS:
            continue

        # Z-score the spike counts
        spike_counts = (spike_counts - np.mean(spike_counts, axis=0)) / np.std(spike_counts, axis=0)
        n_neurons = spike_counts.shape[1]

        # Create arrays per context (trials x spatial bins x neurons)
        context_A = spike_counts[context_per_bin == obj1_goal, :].reshape(
            n_trials_A, n_bins, n_neurons)[:min_trials, :, :]
        context_B = spike_counts[context_per_bin == 3 - obj1_goal, :].reshape(
            n_trials_B, n_bins, n_neurons)[:min_trials, :, :]

        # Fit LDA to part of the environment to fit to (defined by FIT_POS)
        fit_A = np.mean(context_A[:, (unique_positions > FIT_POS[0]) & (unique_positions < FIT_POS[1]), :],
                        axis=1)
        fit_B = np.mean(context_B[:, (unique_positions > FIT_POS[0]) & (unique_positions < FIT_POS[1]), :],
                        axis=1)
        X_train = np.vstack([fit_A, fit_B])
        y_train = np.concatenate([np.ones(min_trials), np.full(min_trials, 2)])
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        lda.fit(X_train, y_train)

        # Project all spatial bins onto the context projection
        lda_dist = np.zeros(unique_positions.shape[0])
        for i, pos in enumerate(unique_positions):

            # Get neural activity for this spatial bin
            bin_A = context_A[:, i, :]
            bin_B = context_B[:, i, :]

            # Project onto LDA axis
            proj_A = lda.transform(bin_A).ravel()
            proj_B = lda.transform(bin_B).ravel()

            # Get LDA distance
            lda_dist[i] = (np.abs(np.mean(proj_A) - np.mean(proj_B))
                       / np.sqrt(0.5 * (np.var(proj_A) + np.var(proj_B))))

        # Store results
        results_df = pd.concat([results_df, pd.DataFrame({
            'subject': this_subject,
            'date': date,
            'region': region,
            'position': unique_positions,
            'lda_dist': lda_dist
        })], ignore_index=True)

# %% Plot results
f, axs = plt.subplots(1, 6, figsize=(7, 1.75), dpi=dpi, sharey=True)
for i, region in enumerate(REGION_ORDER):
    region_df = results_df[results_df['region'] == region]
    sns.lineplot(data=region_df, x='position', y='lda_dist', ax=axs[i], errorbar='se',
                 err_kws={'lw': 0})
    axs[i].set(title=REGION_NAMES[i], ylabel='', xlabel='', xlim=[FIT_POS[1], 1300], ylim=[0, 0.8],
               xticks=[FIT_POS[1], 1300], xticklabels=[FIT_POS[1] / 10, 130])
axs[0].set(ylabel='d prime')

sns.despine(trim=True)
plt.tight_layout()