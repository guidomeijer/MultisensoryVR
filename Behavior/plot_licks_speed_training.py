# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:57:35 2023

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from os.path import join, isfile
import pandas as pd
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style,
                            peri_multiple_events_time_histogram)

# Settings
T_BEFORE = 2
T_AFTER = 2
BIN_SIZE = 0.2
SMOOTHING = 0.1
MIN_LICKS = 30
PLOT_SUBJECTS = ['478153', '478154']


# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths(force_sync=False)
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

# Loop over subjects
for i, subject in enumerate(PLOT_SUBJECTS):

    # List sessions
    sessions = np.array(os.listdir(join(data_path, 'Subjects', subject)))

    """
    # Discard sessions that don't have licks
    lick_ses = np.zeros(len(sessions)).astype(int)
    for j, ses in enumerate(sessions):
        if isfile(join(data_path, 'Subjects', subject, ses, 'lick.times.npy')):
            lick_ses[j] = 1
    sessions = np.array(sessions)[lick_ses.astype(bool)]
    """

    # Select training sessions
    ses_date = np.array([datetime.datetime.strptime(i[:8], '%Y%m%d').date() for i in sessions])
    sessions = sessions[ses_date < subjects.loc[subjects['SubjectID'] == subject,
                                                 'DateFinalTask'].values[0]]
    if len(sessions) == 0:
        continue
    if len(sessions) > 12:
        sessions = sessions[-12:]

    # Create lick figure
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4, figsize=(7,  2*np.ceil(len(sessions) / 4)),
                          dpi=dpi)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    max_y = np.empty(len(sessions))
    for j, ses in enumerate(sessions):

        # Load in data
        trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'))
        lick_times = np.load(join(data_path, 'Subjects', subject, ses, 'lick.times.npy'))
        if lick_times.shape[0] < MIN_LICKS:
            continue

        # Plot
        all_obj_enters = np.concatenate(
            (trials['enterObj1Time'], trials['enterObj2Time'], trials['enterObj3Time']))
        all_obj_ids = np.concatenate(
            (np.ones(trials['enterObj1Time'].shape),
             np.ones(trials['enterObj2Time'].shape)*2,
             np.ones(trials['enterObj3Time'].shape)*3))
        all_obj_ids = all_obj_ids[~np.isnan(all_obj_enters)]
        all_obj_enters = all_obj_enters[~np.isnan(all_obj_enters)]
        
        peri_multiple_events_time_histogram(
            lick_times, np.ones(lick_times.shape[0]), all_obj_enters, all_obj_ids,
            [1], t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, smoothing=SMOOTHING, ax=axs[j],
            pethline_kwargs=[{'color': colors['obj1'], 'lw': 1},
                             {'color': colors['obj2'], 'lw': 1},
                             {'color': colors['obj3'], 'lw': 1}],
            errbar_kwargs=[{'color': colors['obj1'], 'alpha': 0.3, 'lw': 0},
                           {'color': colors['obj2'], 'alpha': 0.3, 'lw': 0},
                           {'color': colors['obj3'], 'alpha': 0.3, 'lw': 0}],
            raster_kwargs=[{'color': colors['obj1'], 'lw': 0.5},
                           {'color': colors['obj2'], 'lw': 0.5},
                           {'color': colors['obj3'], 'lw': 0.5}],
            eventline_kwargs={'lw': 0}, include_raster=True)

        axs[j].set(ylabel='Licks/s', xticks=np.arange(-T_BEFORE, T_AFTER+1, 2),
                   yticks=[0, np.ceil(axs[j].get_ylim()[1])],
                   title=f'{ses} ({trials.shape[0]} trials)', xlabel='')
        axs[j].yaxis.set_label_coords(-0.1, 0.75)
        max_y[j] = axs[j].get_ylim()[1]

    # Place the dotted line now we know the y lim extend
    for j, ses in enumerate(sessions):
        axs[j].plot([0, 0], [0, np.ceil(np.max(max_y))], color='grey', ls='--', lw=0.75, zorder=0)

    f.suptitle(f'{subject}')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)
    if int(np.ceil(len(sessions)/4)) > 1:
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, hspace=0.4)
    else:
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.4)
    # plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], 'Behavior', f'{subject}_reward_zone_entry_licks.jpg'), dpi=600)

    # Create speed figure
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4, figsize=(7, 2*np.ceil(len(sessions) / 4)),
                          dpi=dpi, sharey=True)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    max_y = np.empty(len(sessions))
    for j, ses in enumerate(sessions):

        # Load in data
        trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'))
        wheel_times = np.load(join(data_path, 'Subjects', subject, ses, 'continuous.times.npy'))
        wheel_speed = np.load(join(data_path, 'Subjects', subject,
                              ses, 'continuous.wheelSpeed.npy'))

        # Downsample wheel speed and convert to cm/s
        wheel_speed = wheel_speed[::50] / 10
        wheel_times = wheel_times[::50]

        # Plot
        all_obj_enters = np.concatenate(
            (trials['enterObj1Time'], trials['enterObj2Time'], trials['enterObj3Time']))
        all_obj_ids = np.concatenate(
            (np.ones(trials['enterObj1Time'].shape),
             np.ones(trials['enterObj2Time'].shape)*2,
             np.ones(trials['enterObj3Time'].shape)*3))
        if j == 0:
            peri_event_trace(wheel_speed, wheel_times, all_obj_enters,
                             event_ids=all_obj_ids, color_palette='Set2',
                             event_labels=['1', '2', '3'],
                             t_before=T_BEFORE, t_after=T_AFTER, ax=axs[j], kwargs={'zorder': 1})
        else:
            peri_event_trace(wheel_speed, wheel_times, all_obj_enters,
                             event_ids=all_obj_ids, color_palette='Set2',
                             t_before=T_BEFORE, t_after=T_AFTER, ax=axs[j], kwargs={'zorder': 1})
        axs[j].set(ylabel='Speed (cm/s)', xticks=np.arange(-2, 7, 2), title=f'{ses}', xlabel='')
        max_y = axs[j].get_ylim()[1]

    # Place the dotted line now we know the y lim extend
    for j, ses in enumerate(sessions):
        axs[j].plot([0, 0], [0, np.ceil(np.max(max_y))], color='grey', ls='--', lw=0.75, zorder=0)
        axs[j].set(ylim=[0, np.max(max_y)], yticks=[0, np.ceil(np.max(max_y))])

    f.suptitle(f'{subject}')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)
    if int(np.ceil(len(sessions)/4)) > 1:
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, hspace=0.4)
    else:
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.4)
    plt.savefig(join(path_dict['fig_path'], 'Behavior', f'{subject}_object_entry.jpg'), dpi=600)
