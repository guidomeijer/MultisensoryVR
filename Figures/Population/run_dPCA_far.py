# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:28:35 2025

By Guido Meijer
"""

from ast import If

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy import stats
import mne
import matplotlib.pyplot as plt
from dPCA import dPCA
from msvr_functions import paths, load_subjects, figure_style, load_objects, add_significance
colors, dpi = figure_style()
mne.set_log_level('WARNING')

# Settings
MIN_NEURONS = 5
USE_TYPE = 'ALL'  # INT, PYR or ALL
REGION_ORDER = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']
REGION_NAMES = ['Visual cortex', 'Auditory cortex', 'Temporal association cortex',
                'Perirhinal cortex', 'Lateral entorhinal cortex', 'Hippocampus (CA1)']


def run_stats(df, y_col, t_thres=0.2):
    # Ensure position is numeric so pivoting sorts it correctly/sequentially
    df = df.copy()
    df['position'] = pd.to_numeric(df['position'])
    
    # Average within subject/session/context first
    test1_matrix = df[df['context'] == 1].pivot_table(
        index=['subject', 'session'], columns='position', values=y_col, aggfunc='mean')
    test2_matrix = df[df['context'] == 2].pivot_table(
        index=['subject', 'session'], columns='position', values=y_col, aggfunc='mean')
    
    # CRITICAL: Align the dataframes so rows match perfectly across contexts
    test1_matrix, test2_matrix = test1_matrix.align(test2_matrix, join='inner', axis=0)
    
    if test1_matrix.empty:
        raise ValueError("No matching subject/session pairs found between context 1 and 2.")
        
    positions = test1_matrix.columns.values
    X = test1_matrix.values - test2_matrix.values
    
    # Set threshold: scalar t-threshold for standard clustering, or dict for TFCE
    if isinstance(t_thres, dict):
        threshold = t_thres
    else:
        threshold = stats.t.ppf(1 - t_thres / 2, test1_matrix.shape[0] - 1)
    
    # Run MNE cluster test
    t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        X,
        threshold=threshold,
        n_permutations=1000, 
        tail=0,          
        out_type='mask'  
    )
    
    # Initialize with 1.0, but assign ALL calculated cluster p-values to see reality
    p_values = np.ones(len(positions))
    for cluster_mask, p_val in zip(clusters, cluster_p_values):
        # Assign the actual cluster p-value to its corresponding positions
        p_values[cluster_mask] = p_val
        
    return p_values, positions


# Load in data
path_dict = paths()
subjects = load_subjects()    
with open(path_dict['google_drive_data_path'] / 'residuals_position_50mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Add neuron type to spike_dict
neuron_type = pd.read_csv(path_dict['save_path'] / 'waveform_metrics.csv',
                          dtype={'subject': str, 'date': str})

# Loop over recordings
dpca_df = pd.DataFrame()
dot_df = pd.DataFrame()
for i in np.arange(len(spike_dict['date'])):

    # Get session info
    this_subject = spike_dict['subject'][i]
    this_ses = spike_dict['date'][i]
    this_probe = spike_dict['probe'][i]

    # Subselect neuron types
    if USE_TYPE != 'ALL':
        neuron_types = neuron_type[(neuron_type['subject'] == this_subject) &
                                   (neuron_type['date'] == this_ses) &
                                   (neuron_type['probe'] == this_probe) &
                                   (neuron_type['unit_id'].isin(spike_dict['neuron_id'][i]))]['neuron_type'].values
    else:
        neuron_types = np.array(['ALL'] * spike_dict['neuron_id'][i].shape[0])

    # Get whether this is a FAR or NEAR session
    is_far = subjects.loc[subjects['SubjectID'] == this_subject, 'Far'].values[0]
    
    # Load in object data
    obj_df = load_objects(this_subject, this_ses)

    # Get which context is the rewarded context for the first and second object
    obj1_goal = obj_df.loc[(obj_df['object'] == 1) & (obj_df['goal'] == 1), 'sound'].values[0]
    obj2_goal = obj_df.loc[(obj_df['object'] == 2) & (obj_df['goal'] == 1), 'sound'].values[0]

    # Loop over regions
    unique_regions = np.unique(spike_dict['region'][i])
    for region in unique_regions:
        if region == 'root':
            continue

        # Get data from this session and region
        spike_counts = spike_dict['residuals'][i][
            :, (spike_dict['region'][i] == region) & (neuron_types == USE_TYPE)]  # spatial bins x neurons

        # Throw out silent neurons
        spike_counts = spike_counts[:, np.std(spike_counts, axis=0) > 0.5]

        # If not enough neurons, continue
        if spike_counts.shape[1] < MIN_NEURONS:
            continue
        
        # Z-score the spike counts
        spike_counts = (spike_counts - np.mean(spike_counts, axis=0)) / np.std(spike_counts, axis=0)

        # Get the spatial bins and context for this session
        spatial_bins = spike_dict['position'][i]
        context_per_bin = spike_dict['context'][i]

        # Create a arrays per context (neurons x spatial bins x trials)
        n_bins = np.unique(spatial_bins).shape[0]
        n_neurons = spike_counts.shape[1]
        n_trials_A = spatial_bins[context_per_bin == obj1_goal].shape[0] // n_bins
        n_trials_B = spatial_bins[context_per_bin == 3 - obj1_goal].shape[0] // n_bins
        min_trials = np.min([n_trials_A, n_trials_B])  # trim the extra trials
        state_A_trials = spike_counts[context_per_bin == obj1_goal, :].reshape(n_trials_A, n_bins, n_neurons).transpose(2, 1, 0)[:, :, :min_trials]
        state_B_trials = spike_counts[context_per_bin == 3 - obj1_goal, :].reshape(n_trials_B, n_bins, n_neurons).transpose(2, 1, 0)[:, :, :min_trials]
        
        # Create a 4D array (neurons x state x spatial bins x trials)
        X_trials = np.zeros((n_neurons, 2, n_bins, min_trials))
        X_trials[:, 0, :, :] = state_A_trials
        X_trials[:, 1, :, :] = state_B_trials

        # Create the 3D trial-averaged array (N_neurons, N_states, N_spatial_bins)
        X_mean = np.nanmean(X_trials, axis=-1) 

        # Initialize dPCA
        dpca = dPCA.dPCA(labels='st', regularizer=0.01)
        dpca.protect = ['t'] # Protect the spatial dimension from being mixed

        # Fit the model using BOTH the 3D and 4D arrays
        Z = dpca.fit_transform(X_mean, X_trials)

        # Ensure Context 1 is always the positive branch relative to Context 2
        if np.mean(Z['s'][0, 0, :]) < np.mean(Z['s'][0, 1, :]):
            Z['s'] = -Z['s']
            Z['st'] = -Z['st']

        # Calculate dot product
        w_spatial = dpca.D['t'][:, 0]
        w_context = dpca.D['s'][:, 0]
        dot_prod = np.dot(w_spatial, w_context) / (np.linalg.norm(w_spatial) * np.linalg.norm(w_context))

        dot_df = pd.concat((dot_df, pd.DataFrame(data={
            'subject': [this_subject],
            'session': [this_ses],
            'region': [region],
            'is_far': [is_far],
            'dot_product': [dot_prod],
            'abs_dot_product': [np.abs(dot_prod)]})))

        # Put results in dataframe
        dpca_df = pd.concat((dpca_df, pd.DataFrame(data={
            'pos_traj': np.concatenate((Z['t'][0, 0, :], Z['t'][0, 1, :])),
            'context_traj': np.concatenate((Z['s'][0, 0, :], Z['s'][0, 1, :])),
            'interaction_traj': np.concatenate((Z['st'][0, 0, :], Z['st'][0, 1, :])),
            'context': np.concatenate((np.ones(n_bins), 2*np.ones(n_bins))).astype(int),
            'position': np.tile(np.unique(spatial_bins), 2),
            'subject': this_subject, 
            'session': this_ses, 
            'region': region, 
            'is_far': is_far})))

# Save dot product results
dot_df.to_csv(path_dict['save_path'] / f'dpca_spatial_context_dot_product_{USE_TYPE}.csv', index=False)


# %% Plot
f, axs = plt.subplots(3, 6, figsize=(7, 4.5), dpi=dpi, sharex=True, sharey='row')
for i, region in enumerate(REGION_ORDER):
    plot_df = dpca_df[(dpca_df['region'] == region) & (dpca_df['is_far'] == 1)]

    # Position trajectory
    p_values, positions = run_stats(plot_df, y_col='pos_traj', t_thres=0.2)
    sns.lineplot(data=plot_df, x='position', y='pos_traj', hue='context',
                 hue_order=[1, 2], ax=axs[0, i], palette=[colors['context1'], colors['context2']],
                 legend=False, errorbar='se', err_kws={'lw': 0})
    add_significance(positions, p_values, ax=axs[0, i], y_pos=1.5)
    axs[0, i].set(ylabel='', xlabel='', ylim=[-1.5, 1.5], yticks=[-1.5, 0, 1.5], yticklabels=[-1.5, 0, 1.5],
                  title=REGION_NAMES[i])

    # Context trajectory
    p_values, positions = run_stats(plot_df, y_col='context_traj', t_thres=0.2)
    sns.lineplot(data=plot_df, x='position', y='context_traj', hue='context',
                 ax=axs[1, i], hue_order=[1, 2], palette=[colors['context1'], colors['context2']],
                 legend=False, errorbar='se', err_kws={'lw': 0})
    #sns.lineplot(data=plot_df, x='position', y='context_traj', hue='context',
    #             ax=axs[1, i], hue_order=[1, 2], palette=[colors['context1'], colors['context2']],
    #             legend=False, units='session', estimator=None)
    add_significance(positions, p_values, ax=axs[1, i], y_pos=0.4)
    axs[1, i].set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150], ylabel='', xlabel='',
                  ylim=[-0.4, 0.4], yticks=[-0.4, 0, 0.4], yticklabels=[-0.4, 0, 0.4])

    # Interaction trajectory
    p_values, positions = run_stats(plot_df, y_col='interaction_traj', t_thres=0.2)
    sns.lineplot(data=plot_df, x='position', y='interaction_traj', hue='context',
                 hue_order=[1, 2], ax=axs[2, i], palette=[colors['context1'], colors['context2']],
                 legend=False, errorbar='se', err_kws={'lw': 0}, zorder=1)
    add_significance(positions, p_values, ax=axs[2, i])
    axs[2, i].plot([425, 425], [-1.5, 1.5], ls='--', lw=0.5, color='k', zorder=0)
    axs[2, i].plot([1325, 1325], [-1.5, 1.5], ls='--', lw=0.5, color='k', zorder=0)
    axs[2, i].plot([0, 1500], [0, 0], ls='--', lw=0.5, color='k', zorder=0)
    axs[2, i].set(xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
                  ylim=[-1.5, 1.5], yticks=[-1.5, 0, 1.5], yticklabels=[-1.5, 0, 1.5],
                  xlabel='', ylabel='')

axs[0, 0].set_ylabel('Position trajectory', labelpad=0)
axs[1, 0].set_ylabel('Context trajectory', labelpad=0)
axs[2, 0].set_ylabel('Context-Space interaction', labelpad=0)
f.supxlabel('Position (cm)', fontsize=7, y=0.03)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'dPCA' / f'dpca_far_trajectories_{USE_TYPE}.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'dPCA' / f'dpca_far_trajectories_{USE_TYPE}.jpg', dpi=600)
plt.show()
