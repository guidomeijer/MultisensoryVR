# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:32:45 2023

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from os.path import join, isfile, isdir
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from brainbox.plot import peri_event_time_histogram
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style,
                            peri_multiple_events_time_histogram, load_neural_data)

# Settings
SUBJECT = '459602'
DATE = '20240318'
PROBE = 'probe00'
T_BEFORE = 0.5
T_AFTER = 1

# Get paths
path_dict = paths(sync=False)

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE, histology=False, only_good=True)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))
wheel_time = np.load(join(path_dict['local_data_path'],
                     'Subjects', SUBJECT, DATE, 'continuous.times.npy'))
wheel_dist = np.load(join(path_dict['local_data_path'], 'Subjects',
                     SUBJECT, DATE, 'continuous.wheelDistance.npy'))
wheel_speed = np.load(join(path_dict['local_data_path'],
                      'Subjects', SUBJECT, DATE, 'continuous.wheelSpeed.npy'))
reward_times = np.load(join(path_dict['local_data_path'],
                       'Subjects', SUBJECT, DATE, 'reward.times.npy'))

"""
# Select IBL good neurons
use_neurons = np.where(clusters['ibl_label'] == 1)[0]
spikes['times'] = spikes['times'][np.isin(spikes['clusters'], use_neurons)]
spikes['distances'] = spikes['distances'][np.isin(spikes['clusters'], use_neurons)]
spikes['clusters'] = spikes['clusters'][np.isin(spikes['clusters'], use_neurons)]
"""

# Get relative distance in each trial
trial_dist, dist_time = [], []
trial_start_dist = np.empty(trials.shape[0])
for t in trials.index:
    this_dist = wheel_dist[((wheel_time > trials.loc[t, 'enterEnvTime'])
                            & (wheel_time < trials.loc[t, 'exitEnvTime']))]
    this_time = wheel_time[((wheel_time > trials.loc[t, 'enterEnvTime'])
                            & (wheel_time < trials.loc[t, 'exitEnvTime']))]
    trial_dist.append(this_dist)
    dist_time.append(this_time)
    trial_start_dist[t] = this_dist[0]
trial_dist = np.concatenate(trial_dist)
dist_time = np.concatenate(dist_time)

# Plot neurons
if not isdir(join(path_dict['fig_path'], 'ExampleNeurons', 'PlaceActivity',
            f'{SUBJECT}')):
    os.mkdir(join(path_dict['fig_path'], 'ExampleNeurons', 'PlaceActivity',
                  f'{SUBJECT}'))
if not isdir(join(path_dict['fig_path'], 'ExampleNeurons', 'PlaceActivity',
            f'{SUBJECT}', f'{DATE}')):
    os.mkdir(join(path_dict['fig_path'], 'ExampleNeurons', 'PlaceActivity',
                  f'{SUBJECT}', f'{DATE}'))
colors, dpi = figure_style()
for i, neuron_id in enumerate(np.unique(spikes['clusters'])):

    # Plot spatial firing
    spike_times = spikes['times'][spikes['clusters'] == neuron_id]
    spike_dist = spikes['distances'][spikes['clusters'] == neuron_id]

    # Plot spatial firing PSTH
    try:
        f, ax = plt.subplots(figsize=(2, 2.5), dpi=600)
        peri_event_time_histogram(spike_dist / 10, np.ones(spike_dist.shape), trial_start_dist / 10, [1],
                                  include_raster=True, error_bars='sem', ax=ax,
                                  t_before=0, t_after=150, smoothing=2, bin_size=1,
                                  pethline_kwargs={'color': 'black', 'lw': 1},
                                  errbar_kwargs={'color': 'black', 'alpha': 0.3, 'lw': 0},
                                  raster_kwargs={'color': 'black', 'lw': 0.3},
                                  eventline_kwargs={'lw': 0})
        
        ax.plot([-2, -2], [0, 1], color='k', lw=0.75, clip_on=False)
        ax.text(-2, 1/2, '1 sp cm$^{-1}$', ha='right', va='center', rotation=90)
        ax.text(-2, ax.get_ylim()[0] / 2, f'{trial_start_dist.shape[0]} trials',
                ha='right', va='center', rotation=90)
        ax.spines[['left']].set_visible(False)
        ax.set(xlabel='Distance (cm)', xticks=[0, 50, 100, 150], ylabel='', yticks=[])
        plt.tight_layout()

        plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', 'PlaceActivity',
                    f'{SUBJECT}', f'{DATE}', f'{PROBE}_neuron{neuron_id}.jpg'))
        plt.close(f)
    except:
        continue
