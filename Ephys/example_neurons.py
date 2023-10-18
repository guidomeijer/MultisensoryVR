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
from os.path import join, isfile
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from brainbox.plot import peri_event_time_histogram
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style,
                            peri_multiple_events_time_histogram, load_spikes)

# Settings
SUBJECT = '450409'
DATE = '20231012'
PROBE = 'probe00'
T_BEFORE = 0.5
T_AFTER = 1

# Get paths
path_dict = paths()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters = load_spikes(session_path, PROBE, only_bc_good=False)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))
wheel_time = np.load(join(path_dict['local_data_path'],
                     'Subjects', SUBJECT, DATE, 'continuous.times.npy'))
wheel_dist = np.load(join(path_dict['local_data_path'], 'Subjects',
                     SUBJECT, DATE, 'continuous.wheelDistance.npy'))
wheel_speed = np.load(join(path_dict['local_data_path'],
                      'Subjects', SUBJECT, DATE, 'continuous.wheelSpeed.npy'))
reward_times = np.load(join(path_dict['local_data_path'],
                       'Subjects', SUBJECT, DATE, 'reward.times.npy'))

# Plot neurons
colors, dpi = figure_style()
for i, neuron_id in enumerate(np.unique(spikes['clusters'])):

    # Plot reward timing PSTH
    f, ax = plt.subplots(figsize=(2, 2.5), dpi=600)
    try:
        peri_event_time_histogram(spikes['times'], spikes['clusters'], reward_times, neuron_id,
                                  include_raster=True, error_bars='sem', ax=ax,
                                  t_before=T_BEFORE, t_after=T_AFTER,
                                  pethline_kwargs={'color': 'black', 'lw': 1},
                                  errbar_kwargs={'color': 'black', 'alpha': 0.3, 'lw': 0},
                                  raster_kwargs={'color': 'black', 'lw': 0.3},
                                  eventline_kwargs={'lw': 0})
    except:
        continue
    ax.set(ylabel='Firing rate (spks/s)', xlabel='Time from reward delivery (s)',
           yticks=[np.round(ax.get_ylim()[1])], xticks=[-0.5, 0, 0.5, 1],
           ylim=[ax.get_ylim()[0], np.round(ax.get_ylim()[1])])
    # ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', 'RewardDelivery',
                f'{SUBJECT}_{DATE}_{PROBE}_neuron{neuron_id}.jpg'))
    plt.close(f)

    # Plot spatial firing
    f, ax = plt.subplots(figsize=(2, 2.5), dpi=600)
    peri_event_time_histogram(spikes['distances'], spikes['clusters'], trials['enterEnvDist'],
                              neuron_id,
                              include_raster=True, error_bars='sem', ax=ax,
                              t_before=0, t_after=800, smoothing=20, bin_size=20,
                              pethline_kwargs={'color': 'black', 'lw': 1},
                              errbar_kwargs={'color': 'black', 'alpha': 0.3, 'lw': 0},
                              raster_kwargs={'color': 'black', 'lw': 0.3},
                              eventline_kwargs={'lw': 0})
    ax.set(ylabel='Firing rate (spks/cm)', xlabel='Distance (cm)',
           yticks=[np.round(ax.get_ylim()[1], decimals=2)], xticks=[0, 400, 800],
           ylim=[ax.get_ylim()[0], np.round(ax.get_ylim()[1], decimals=2)])
    plt.tight_layout()
    
    plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', 'PlaceActivity',
                f'{SUBJECT}_{DATE}_{PROBE}_neuron{neuron_id}.jpg'))
    plt.close(f)
