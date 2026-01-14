# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import torch
import seaborn as sns
from zetapy import zetatstest2
import matplotlib.pyplot as plt
from msvr_functions import (paths, load_multiple_probes, load_objects, combine_regions,
                            peri_event_trace, figure_style, N_PATTERNS)
from ppseq.plotting import plot_model, color_plot
from ppseq.model import PPSeq
colors, dpi = figure_style()

# Settings
MIN_NEURONS = 5
BIN_WIDTH = 0.15
SIG_TIME = 2  # s

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])

pattern_df = pd.DataFrame()
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
        time_ax = np.arange(0, region_spikes.max(), BIN_WIDTH) + BIN_WIDTH / 2
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
                      template_duration=10,
                      alpha_a0=1.5,
                      beta_a0=0.2,
                      alpha_b0=1,
                      beta_b0=0.1,
                      alpha_t0=1.2,
                      beta_t0=0.1)
        lps, amplitudes_torch = model.fit(data, num_iter=100)
        amplitudes = amplitudes_torch.cpu().numpy()

        # Find patterns that encode reward prediction
        obj1_goal = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 1), 'times'].values
        obj1_nogoal = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 0), 'times'].values
        obj2_goal = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 1), 'times'].values
        obj2_nogoal = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 0), 'times'].values
        p_obj1, p_obj2 = np.empty(amplitudes.shape[0]), np.empty(amplitudes.shape[0])
        for pat in range(amplitudes.shape[0]):

            # Run zeta test (make sure not to include the bin that can overlap with 0)
            p_obj1[pat], _ = zetatstest2(time_ax, amplitudes[pat, :], obj1_goal - SIG_TIME,
                                         time_ax, amplitudes[pat, :], obj1_nogoal - SIG_TIME,
                                         dblUseMaxDur=SIG_TIME - (BIN_WIDTH/2))
            p_obj2[pat], _ = zetatstest2(time_ax, amplitudes[pat, :], obj2_goal - SIG_TIME,
                                         time_ax, amplitudes[pat, :], obj2_nogoal - SIG_TIME,
                                         dblUseMaxDur=SIG_TIME - (BIN_WIDTH/2))

        # Plot
        f, axs = plt.subplots(2, N_PATTERNS[region], figsize=(1.75*N_PATTERNS[region], 3.5),
                              sharex=True, dpi=dpi)
        for pp in range(N_PATTERNS[region]):
            peri_event_trace(amplitudes[pp, :], time_ax, all_obj_df.loc[all_obj_df['object'] == 1, 'times'],
                             all_obj_df.loc[all_obj_df['object'] == 1, 'goal'].values + 1,
                             t_before=3, t_after=1, ax=axs[0, pp],
                             color_palette=[colors['no-goal'], colors['goal']])
            axs[0, pp].set(xticks=np.arange(-3, 1.5), ylabel='Pattern activation', title='First object')
        for pp in range(N_PATTERNS[region]):
            peri_event_trace(amplitudes[pp, :], time_ax, all_obj_df.loc[all_obj_df['object'] == 2, 'times'],
                             all_obj_df.loc[all_obj_df['object'] == 2, 'goal'].values + 1,
                             t_before=3, t_after=1, ax=axs[1, pp],
                             color_palette=[colors['no-goal'], colors['goal']])
            axs[1, pp].set(xticks=np.arange(-3, 1.5), ylabel='Pattern activation',
                           xlabel='Time from object entry (s)', title='Second object')
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns' / f'{region}_{subject}_{date}_reward.jpg', dpi=600)
        plt.close(f)

        # Find patterns that respond to sound onset
        sound1 = trials.loc[trials['soundId'] == 1, 'soundOnsetTime'].values
        sound2 = trials.loc[trials['soundId'] == 2, 'soundOnsetTime'].values
        p_sound = np.empty(amplitudes.shape[0])
        for pat in range(amplitudes.shape[0]):

            # Run zeta test (make sure not to include the bin that can overlap with 0)
            p_sound[pat], _ = zetatstest2(time_ax, amplitudes[pat, :], sound1,
                                          time_ax, amplitudes[pat, :], sound2,
                                          dblUseMaxDur=SIG_TIME)

        # Plot
        f, axs = plt.subplots(1, N_PATTERNS[region], figsize=(1.75*N_PATTERNS[region], 1.75), dpi=dpi)
        for pp in range(N_PATTERNS[region]):
            peri_event_trace(amplitudes[pp, :], time_ax, trials['soundOnsetTime'], trials['soundId'],
                             t_before=1, t_after=3, ax=axs[pp],
                             color_palette=[colors['sound1'], colors['sound2']])
            axs[pp].set(xticks=np.arange(-1, 3.5), ylabel='Pattern activation')
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns' / f'{region}_{subject}_{date}_sound.jpg', dpi=600)
        plt.close(f)

        # Add to df
        pattern_df = pd.concat((pattern_df, pd.DataFrame(data={
            'p_obj1': p_obj1, 'p_obj2': p_obj2, 'p_sound_id': p_sound,
            'pattern': np.arange(1, N_PATTERNS[region]+1),
            'region': region, 'subject': subject, 'date': date
            })))

    # Save to disk
    pattern_df.to_csv(path_dict['save_path'] / 'spike_pattern_sig.csv', index=False)

