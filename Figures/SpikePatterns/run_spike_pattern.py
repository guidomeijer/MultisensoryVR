# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy import stats, signal
import matplotlib.pyplot as plt
from msvr_functions import (paths, load_multiple_probes, load_objects, combine_regions,
                            peri_event_trace, figure_style, N_PATTERNS, plot_ppseq)
from ppseq.model import PPSeq
colors, dpi = figure_style()

# Settings
MIN_NEURONS = 5
BIN_WIDTH = 0.01
SIG_TIME = 1  # s
OVERWRITE = True
PLOT = True
MIN_RIPPLES = 20

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)

if OVERWRITE:
    pattern_df = pd.DataFrame()
else:
    pattern_df = pd.read_csv(path_dict['save_path'] / 'spike_pattern_sig.csv')
    pattern_df['subject'] = pattern_df['subject'].astype(str)
    pattern_df['date'] = pattern_df['date'].astype(str)
    pattern_keys = pattern_df[['subject', 'date']].drop_duplicates()
    rec_merged = rec.merge(pattern_keys, on=['subject', 'date'], how='left', indicator=True)
    rec = rec_merged[rec_merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    rec = rec.reset_index(drop=True)

#%% Functions
def get_windowed_means(event_times, time_ax, signal, window=(-1, 0)):
    """
    Extracts the mean signal amplitude within a specific window relative to event times.
    """
    event_means = []
    for t in event_times:
        # Define the absolute time boundaries
        t_start = t + window[0]
        t_end = t + window[1]
        
        # Find indices within the time window
        mask = (time_ax >= t_start) & (time_ax < t_end)
        
        if np.any(mask):
            event_means.append(np.mean(signal[mask]))
        else:
            # Handle cases where the window might be outside the recorded time_ax
            event_means.append(np.nan)
            
    return np.array(event_means)



#%% MAIN
for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'\nStarting {subject} {date} [{i} of {rec.shape[0]}]')

    # Load in data
    session_path = path_dict['local_data_path'] / 'subjects' / f'{subject}' / f'{date}'
    trials = pd.read_csv(session_path / 'trials.csv')
    spikes, clusters, channels = load_multiple_probes(session_path, min_fr=0.5)
    all_obj_df = load_objects(subject, date)

    # Merge regions
    for probe in clusters.keys():
        clusters[probe]['region'] = combine_regions(clusters[probe]['acronym'], split_peri=False)

    # Get list of all regions and which probe they were recorded on
    regions, region_probes = [], []
    for p, probe in enumerate(spikes.keys()):
        unique_regions = np.unique(clusters[probe]['region'])
        unique_regions = unique_regions[unique_regions != 'root']
        if (probe == 'probe00') & ('CA1' in unique_regions):
            unique_regions = unique_regions[unique_regions != 'CA1']
        regions.append(unique_regions)
        region_probes.append([probe] * unique_regions.shape[0])
    regions = np.concatenate(regions)
    region_probes = np.concatenate(region_probes)

    # Loop over all brain regions
    for region, probe in zip(regions, region_probes):
        print(f'{region}')

        # Get spiking activity from this region
        region_neuron_ids = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
        if (region_neuron_ids.shape[0] < MIN_NEURONS) or (region == 'root'):
            continue
        region_spikes = spikes[probe]['times'][np.isin(spikes[probe]['clusters'], region_neuron_ids)]
        region_clusters = spikes[probe]['clusters'][np.isin(spikes[probe]['clusters'], region_neuron_ids)]

        """
        # Select spikes in active period
        region_clusters = region_clusters[((region_spikes > trials['enterEnvTime'].values[0])
                                           & (region_spikes < trials['exitEnvTime'].values[-1]))]
        region_spikes = region_spikes[((region_spikes > trials['enterEnvTime'].values[0])
                                       & (region_spikes < trials['exitEnvTime'].values[-1]))]
        """

        # Create input for PPSeq
        list_of_spiketimes = []
        for n, neuron_id in enumerate(region_neuron_ids):
            list_of_spiketimes.append(region_spikes[region_clusters == neuron_id])
        num_timesteps = int(region_spikes.max() // BIN_WIDTH) + 1
        time_ax = np.arange(num_timesteps) * BIN_WIDTH + BIN_WIDTH / 2
        num_neurons = len(list_of_spiketimes)

        indices = []
        values = []
        for ii, these_spike_times in enumerate(list_of_spiketimes):
            bins = (torch.tensor(these_spike_times, device='cuda') // BIN_WIDTH).long()
            indices.append(torch.stack([torch.full_like(bins, ii), bins]))
            values.append(torch.ones_like(bins, dtype=torch.float32))

        if indices:
            indices = torch.cat(indices, dim=1)
            values = torch.cat(values)
            data = torch.sparse_coo_tensor(indices, values, size=(num_neurons, num_timesteps),
                                           device='cuda').to_dense()
        else:
            data = torch.zeros(num_neurons, num_timesteps, device='cuda')
        data = data[~torch.all(data == 0, dim=1)]

        # Run ppseq
        model = PPSeq(num_templates=N_PATTERNS[region],
                      num_neurons=num_neurons,
                      template_duration=20,
                      alpha_a0=1.5,
                      beta_a0=1,
                      alpha_b0=1,
                      beta_b0=0.1,
                      alpha_t0=1.2,
                      beta_t0=0.1)
        lps, amplitudes_torch = model.fit(data, num_iter=200)
        amplitudes = amplitudes_torch.cpu().numpy()
        templates = model.templates.cpu()

        # Convolve the spike pattern events with the pattern's temporal profile
        activation_rate = np.zeros_like(amplitudes)
        for pattern_idx in range(amplitudes.shape[0]):
            
            # Summing across neurons collapses the spatial dimension, leaving just the time course
            pattern_profile = templates[pattern_idx].sum(axis=0) 
            
            # 2. Reconstruct the Continuous Activation Trace (Convolution)
            # This converts sparse impulses into the actual shape of the activity
            amp_series = amplitudes[pattern_idx, :]
            continuous_rate = signal.convolve(amp_series, pattern_profile, mode='full')
            activation_rate[pattern_idx, :] = continuous_rate[:len(amp_series)]
            

        if PLOT:
            # Plot patterns
            f, ax = plot_ppseq(data.cpu()[:, -5000:-4000], model, amplitudes_torch.cpu()[:, -5000:-4000])
            ax.set(xticks=[0, 33, 66, 100], xticklabels=[0, 5, 10, 15], yticks=[],
                   ylabel='Sorted neurons', xlabel='Time (s)')
            sns.despine(trim=True, left=True)
            plt.tight_layout()
            plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns' / f'{region}_{subject}_{date}_example.jpg', dpi=600)
            plt.close(f)

        # Find patterns that encode reward prediction
        obj1_goal = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 1), 'times'].values
        obj1_nogoal = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 0), 'times'].values
        obj2_goal = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 1), 'times'].values
        obj2_nogoal = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 0), 'times'].values
        p_obj1, p_obj2 = np.empty(activation_rate.shape[0]), np.empty(activation_rate.shape[0])
        t_obj1, t_obj2 = np.empty(activation_rate.shape[0]), np.empty(activation_rate.shape[0])
        for pat in range(activation_rate.shape[0]):
            
            # Object 1
            goal_samples = get_windowed_means(obj1_goal, time_ax, activation_rate[pat, :], window=(-SIG_TIME, 0))
            nogoal_samples = get_windowed_means(obj1_nogoal, time_ax, activation_rate[pat, :], window=(-SIG_TIME, 0))
            t_obj1[pat], p_obj1[pat] = stats.ttest_ind(goal_samples, nogoal_samples)
            
            # Object 2
            goal_samples = get_windowed_means(obj2_goal, time_ax, activation_rate[pat, :], window=(-SIG_TIME, 0))
            nogoal_samples = get_windowed_means(obj2_nogoal, time_ax, activation_rate[pat, :], window=(-SIG_TIME, 0))
            t_obj2[pat], p_obj2[pat] = stats.ttest_ind(goal_samples, nogoal_samples)
            
        
        if PLOT:
            # Plot
            f, axs = plt.subplots(2, N_PATTERNS[region], figsize=(1.75*N_PATTERNS[region], 3.5),
                                  sharex=True, dpi=dpi)
            for pp in range(N_PATTERNS[region]):
                peri_event_trace(activation_rate[pp, :], time_ax, all_obj_df.loc[all_obj_df['object'] == 1, 'times'],
                                 all_obj_df.loc[all_obj_df['object'] == 1, 'goal'].values + 1,
                                 t_before=3, t_after=1, ax=axs[0, pp],
                                 color_palette=[colors['no-goal'], colors['goal']])
                axs[0, pp].set(xticks=np.arange(-3, 1.5), ylabel='Pattern activation',
                               title=f'p={np.round(p_obj1[pp], 3)}')
            for pp in range(N_PATTERNS[region]):
                peri_event_trace(activation_rate[pp, :], time_ax, all_obj_df.loc[all_obj_df['object'] == 2, 'times'],
                                 all_obj_df.loc[all_obj_df['object'] == 2, 'goal'].values + 1,
                                 t_before=3, t_after=1, ax=axs[1, pp],
                                 color_palette=[colors['no-goal'], colors['goal']])
                axs[1, pp].set(xticks=np.arange(-3, 1.5), ylabel='Pattern activation',
                               xlabel='Time from object entry (s)', title=f'p={np.round(p_obj2[pp], 3)}')
            sns.despine(trim=True)
            plt.tight_layout()
                        
            plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns' / f'{region}_{subject}_{date}_reward.jpg', dpi=600)
            plt.close(f)

        # Find patterns that respond to sound onset
        sound1 = trials.loc[trials['soundId'] == 1, 'soundOnsetTime'].values
        sound2 = trials.loc[trials['soundId'] == 2, 'soundOnsetTime'].values
        p_sound, t_sound = np.empty(activation_rate.shape[0]), np.empty(activation_rate.shape[0])
        for pat in range(activation_rate.shape[0]):

            # Test for difference
            goal_samples = get_windowed_means(sound1, time_ax, activation_rate[pat, :], window=(-SIG_TIME, 0))
            nogoal_samples = get_windowed_means(sound2, time_ax, activation_rate[pat, :], window=(-SIG_TIME, 0))
            p_sound[pat], t_sound[pat] = stats.ttest_ind(goal_samples, nogoal_samples)
          
        if PLOT:
            # Plot
            f, axs = plt.subplots(1, N_PATTERNS[region], figsize=(1.75*N_PATTERNS[region], 1.75), dpi=dpi)
            for pp in range(N_PATTERNS[region]):
                peri_event_trace(activation_rate[pp, :], time_ax, trials['soundOnsetTime'], trials['soundId'],
                                 t_before=1, t_after=3, ax=axs[pp],
                                 color_palette=[colors['sound1'], colors['sound2']])
                axs[pp].set(xticks=np.arange(-1, 3.5), ylabel='Pattern activation')
            sns.despine(trim=True)
            plt.tight_layout()
            plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns' / f'{region}_{subject}_{date}_sound.jpg', dpi=600)
            plt.close(f)
            
        # Find which spike patterns are active during ripples
        these_ripples = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
        p_ripples, t_ripples = np.full(activation_rate.shape[0], np.nan), np.full(activation_rate.shape[0], np.nan)
        if these_ripples.shape[0] >= MIN_RIPPLES:
            
            # Drop last ripple if too close to the end of the recording
            if time_ax[-1] - these_ripples['start_times'].values[-1] < SIG_TIME:
                these_ripples = these_ripples[:-1]   
            """
            # Get significance per spike pattern
            for pat in range(amplitudes.shape[0]):
                
                
                p_ripples[pat], ZETA = zetatstest(time_ax, amplitudes[pat, :],
                                                  these_ripples['start_times'].values - (SIG_TIME/2),
                                                  dblUseMaxDur=SIG_TIME)
                z_ripples[pat] = ZETA['dblZETADeviation']
            """
                
            if PLOT:
                # Plot
                f, axs = plt.subplots(1, N_PATTERNS[region], figsize=(1.75*N_PATTERNS[region], 1.75), dpi=dpi)
                for pp in range(N_PATTERNS[region]):
                    peri_event_trace(activation_rate[pp, :], time_ax, these_ripples['start_times'],
                                     np.ones(these_ripples.shape[0]),
                                     t_before=1, t_after=1, ax=axs[pp])
                    axs[pp].set(xticks=[-1, 0, 1], ylabel='Pattern activation')
                sns.despine(trim=True)
                plt.tight_layout()
                plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns' / f'{region}_{subject}_{date}_ripples.jpg', dpi=600)
                plt.close(f)        
        
        # Add to df
        pattern_df = pd.concat((pattern_df, pd.DataFrame(data={
            'p_obj1': p_obj1, 'p_obj2': p_obj2, 'p_sound_id': p_sound, 'p_ripples': p_ripples,
            't_obj1': t_obj1, 't_obj2': t_obj2, 't_sound_id': t_sound, 'z_ripples': t_ripples,
            'pattern': np.arange(1, N_PATTERNS[region]+1),
            'region': region, 'subject': subject, 'date': date
            })))
        
        # Save spike pattern amplitudes to disk
        np.save(path_dict['google_drive_data_path'] / 'SpikePatterns' / f'{subject}_{date}_{region}.amplitudes.npy',
                amplitudes)
        np.save(path_dict['google_drive_data_path'] / 'SpikePatterns' / f'{subject}_{date}_{region}.times.npy',
                time_ax)
        
    # Save to disk
    pattern_df.to_csv(path_dict['save_path'] / 'spike_pattern_sig.csv', index=False)

