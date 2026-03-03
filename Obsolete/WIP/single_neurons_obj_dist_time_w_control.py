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
                            peri_multiple_events_time_histogram, load_neural_data)

# Settings
SUBJECT = '459602'
DATE = '20240315'
PROBE = 'probe00'
T_BEFORE = 1
T_AFTER = 2
T_BIN_SIZE = 0.05
T_SMOOTHING = 0.025
DIST_BEFORE = 5
DIST_AFTER = 15
DIST_BIN_SIZE = 0.5
DIST_SMOOTHING = 0.25

# Get paths
path_dict = paths()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE, histology=False, only_good=True)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))
subjects = load_subjects()

# Select couple of trials
#trials = trials[:50]

# Get goal, distractor and control object ids
sound1_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'Sound1Obj'].values[0]
sound2_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'Sound2Obj'].values[0]
control_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'ControlObject'].values[0]

# Get object entry distances
goal_obj_dist = np.concatenate((trials.loc[(trials['soundId'] == sound1_obj), 'enterObj1Pos'],
                                trials.loc[(trials['soundId'] == sound2_obj), 'enterObj2Pos']))
distractor_obj_dist = np.concatenate((trials.loc[(trials['soundId'] == sound2_obj), 'enterObj1Pos'],
                                      trials.loc[(trials['soundId'] == sound1_obj), 'enterObj2Pos']))
control_obj_dist = trials['enterObj3Pos']
all_obj_dist = np.concatenate((goal_obj_dist, distractor_obj_dist, control_obj_dist))
all_obj_dist_ids = np.concatenate((np.ones(goal_obj_dist.shape[0]),
                                   np.ones(distractor_obj_dist.shape[0]) * 2,
                                   np.ones(control_obj_dist.shape[0]) * 3))
sort_idx = np.argsort(all_obj_dist)
all_obj_dist = all_obj_dist[sort_idx]
all_obj_dist_ids = all_obj_dist_ids[sort_idx]

# Get object entry times
goal_obj_times = np.concatenate((trials.loc[(trials['soundId'] == sound1_obj), 'enterObj1'],
                                 trials.loc[(trials['soundId'] == sound2_obj), 'enterObj2']))
distractor_obj_times = np.concatenate((trials.loc[(trials['soundId'] == sound2_obj), 'enterObj1'],
                                       trials.loc[(trials['soundId'] == sound1_obj), 'enterObj2']))
control_obj_times = trials['enterObj3']
all_obj_times = np.concatenate((goal_obj_times, distractor_obj_times, control_obj_times))
all_obj_times_ids = np.concatenate((np.ones(goal_obj_times.shape[0]),
                                    np.ones(distractor_obj_times.shape[0]) * 2,
                                    np.ones(control_obj_times.shape[0]) * 3))
all_obj_times_ids = all_obj_times_ids[~np.isnan(all_obj_times)]
all_obj_times = all_obj_times[~np.isnan(all_obj_times)]
sort_idx = np.argsort(all_obj_times)
all_obj_times = all_obj_times[sort_idx]
all_obj_times_ids = all_obj_times_ids[sort_idx]

# Plot neurons
colors, dpi = figure_style()
for i, neuron_id in enumerate(np.unique(spikes['clusters'])):
    print(f'Plotting neuron {i} of {np.unique(spikes["clusters"]).shape[0]}')

    # Plot object entry 
    try:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2.5), dpi=600)
        peri_multiple_events_time_histogram(spikes['times'], spikes['clusters'],
                                            all_obj_times, all_obj_times_ids, neuron_id,
                                            include_raster=True, error_bars='sem', ax=ax1,
                                            t_before=T_BEFORE, t_after=T_AFTER,
                                            smoothing=T_SMOOTHING, bin_size=T_BIN_SIZE,
                                            pethline_kwargs=[{'color': 'mediumseagreen', 'lw': 1},
                                                             {'color': 'crimson', 'lw': 1},
                                                             {'color': 'slategrey', 'lw': 1}],
                                            errbar_kwargs=[{'color': 'mediumseagreen', 'alpha': 0.3, 'lw': 0},
                                                           {'color': 'crimson', 'alpha': 0.3, 'lw': 0},
                                                           {'color': 'slategrey', 'alpha': 0.3, 'lw': 0}],
                                            raster_kwargs=[{'color': 'mediumseagreen', 'lw': 0.5},
                                                           {'color': 'crimson', 'lw': 0.5},
                                                           {'color': 'slategrey', 'lw': 0.5}],
                                            eventline_kwargs={'lw': 0})
        y_top = np.ceil(ax1.get_ylim()[1])
        if y_top == 0:
            y_top = ax1.get_ylim()[1]
        ax1.set(ylabel='Firing rate (spks/s)', xlabel='Time from object entry (s)',
               yticks=[y_top], 
               ylim=[ax1.get_ylim()[0], y_top])
        # ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        # Plot object time
        peri_multiple_events_time_histogram(spikes['distances'] / 10, spikes['clusters'],
                                            all_obj_dist / 10, all_obj_dist_ids, neuron_id,
                                            include_raster=True, error_bars='sem', ax=ax2,
                                            t_before=DIST_BEFORE, t_after=DIST_AFTER,
                                            smoothing=DIST_SMOOTHING, bin_size=DIST_BIN_SIZE,
                                            pethline_kwargs=[{'color': 'mediumseagreen', 'lw': 1},
                                                             {'color': 'crimson', 'lw': 1},
                                                             {'color': 'slategrey', 'lw': 1}],
                                            errbar_kwargs=[{'color': 'mediumseagreen', 'alpha': 0.3, 'lw': 0},
                                                           {'color': 'crimson', 'alpha': 0.3, 'lw': 0},
                                                           {'color': 'slategrey', 'alpha': 0.3, 'lw': 0}],
                                            raster_kwargs=[{'color': 'mediumseagreen', 'lw': 0.5},
                                                           {'color': 'crimson', 'lw': 0.5},
                                                           {'color': 'slategrey', 'lw': 0.5}],
                                            eventline_kwargs={'lw': 0})
        y_top = np.ceil(ax2.get_ylim()[1])
        if y_top == 0:
            y_top = ax2.get_ylim()[1]
        ax2.set(ylabel='', xlabel='Distance from object entry (cm)',
               yticks=[y_top], 
               ylim=[ax2.get_ylim()[0], y_top])
        # ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        plt.tight_layout()
        plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', 'ObjectEntry',
                    f'{SUBJECT}_{DATE}_{PROBE}_neuron{neuron_id}.jpg'))
        plt.close(f)
    except:
        continue

    
    # Plot sound start    
    trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))
    f, (ax1) = plt.subplots(1, 1, figsize=(3, 2.5), dpi=600)
    peri_multiple_events_time_histogram(spikes['times'], spikes['clusters'],
                                        trials['soundOnset'], trials['soundId'], neuron_id,
                                        include_raster=True, error_bars='sem', ax=ax1,
                                        t_before=T_BEFORE, t_after=5,
                                        smoothing=T_SMOOTHING, bin_size=T_BIN_SIZE,
                                        pethline_kwargs=[{'color': 'mediumturquoise', 'lw': 1},
                                                         {'color': 'peru', 'lw': 1}],
                                        errbar_kwargs=[{'color': 'mediumturquoise', 'alpha': 0.3, 'lw': 0},
                                                       {'color': 'peru', 'alpha': 0.3, 'lw': 0}],
                                        raster_kwargs=[{'color': 'mediumturquoise', 'lw': 0.5},
                                                       {'color': 'peru', 'lw': 0.5}],
                                        eventline_kwargs={'lw': 0})
    y_top = np.ceil(ax1.get_ylim()[1])
    if y_top == 0:
        y_top = ax1.get_ylim()[1]
    ax1.set(ylabel='', xlabel='Time from sound onset (s)',
           yticks=[y_top], 
           ylim=[ax1.get_ylim()[0], y_top])
    # ax.plot([0, 1], [0, 0], lw=2.5, color='royalblue')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', 'SoundStart',
                f'{SUBJECT}_{DATE}_{PROBE}_neuron{neuron_id}.jpg'))
    plt.close(f)
    

 