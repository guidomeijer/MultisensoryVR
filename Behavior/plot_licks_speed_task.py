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
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style, load_objects,
                            peri_multiple_events_time_histogram, load_trials)

# Settings
T_BEFORE = 2
T_AFTER = 3
BIN_SIZE = 0.1
SMOOTHING = 0.1
PLOT_SUBJECTS = ['478153', '478154']

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths(force_sync=True)
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

# Loop over subjects
for i, subject in enumerate(PLOT_SUBJECTS):

    # Get reward contingencies
    sound1_obj = subjects.loc[subjects['SubjectID'] == subject, "Sound1Obj"].values[0]
    sound2_obj = subjects.loc[subjects['SubjectID'] == subject, "Sound2Obj"].values[0]
    control_obj = subjects.loc[subjects['SubjectID'] == subject, "ControlObject"].values[0]    

    # List sessions
    sessions = np.array(os.listdir(join(data_path, 'Subjects', subject)))

    """
    # Discard sessions that don't have licks
    no_lick_ses = np.zeros(len(sessions))
    for j, ses in enumerate(sessions):
        if not isfile(join(data_path, 'Subjects', subject, ses, 'lick.times.npy')):
            no_lick_ses[j] = 1
    sessions = np.array(sessions)[~no_lick_ses.astype(int).astype(bool)]
    """

    # Select final task sessions
    ses_date = np.array([datetime.datetime.strptime(i[:8], '%Y%m%d').date() for i in sessions])
    sessions = sessions[ses_date >= subjects.loc[subjects['SubjectID'] == subject,
                                                 'DateFinalTask'].values[0]]
    if len(sessions) == 0:
        continue
    elif len(sessions) > 8:
        sessions = sessions[-8:]

    # Create lick figure
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4, figsize=(7,  2*np.ceil(len(sessions) / 4)),
                          dpi=dpi)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    max_y = np.empty(len(sessions))
    for j, ses in enumerate(sessions):

        # Load in data
        trials = load_trials(subject, ses)
        obj_df = load_objects(subject, ses)
        lick_times = np.load(join(data_path, 'Subjects', subject, ses, 'lick.times.npy'))
        if lick_times.shape[0] < 100:
            continue
        
        # Get timestamps of entry of goal, no-goal and control object sets
        peri_multiple_events_time_histogram(
            lick_times, np.ones(lick_times.shape[0]),
            obj_df.loc[obj_df['object'] == 1, 'times'].values,
            obj_df.loc[obj_df['object'] == 1, 'goal'].values,
            [1], t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, smoothing=0, ax=axs[j],
            pethline_kwargs=[{'color': colors['no-goal'], 'lw': 1},
                             {'color': colors['goal'], 'lw': 1}],
            errbar_kwargs=[{'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0},
                           {'color': colors['goal'], 'alpha': 0.3, 'lw': 0}],
            raster_kwargs=[{'color': colors['no-goal'], 'lw': 0.5},
                           {'color': colors['goal'], 'lw': 0.5}],
            eventline_kwargs={'lw': 0}, include_raster=True)

        axs[j].set(ylabel='Licks/s', xticks=np.arange(-T_BEFORE, T_AFTER+1),
                   yticks=[0, np.ceil(axs[j].get_ylim()[1])],
                   title=f'{ses} ({trials.shape[0]} trials)', xlabel='')
        axs[j].yaxis.set_label_coords(-0.1, 0.75)
        max_y[j] = axs[j].get_ylim()[1]

    # Place the dotted line now we know the y lim extend
    for j, ses in enumerate(sessions):
        axs[j].plot([0, 0], [0, np.ceil(np.max(max_y))], color='grey', ls='--', lw=0.75, zorder=0)

    f.suptitle(f'{subject} Object 1')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)
    if int(np.ceil(len(sessions)/4)) > 1:
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, hspace=0.4)
    else:
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.8, hspace=0.4)

    # plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], f'{subject}_task_licks_obj1.jpg'), dpi=600)
    
    # Create lick figure
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4, figsize=(7,  2*np.ceil(len(sessions) / 4)),
                          dpi=dpi)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    max_y = np.empty(len(sessions))
    for j, ses in enumerate(sessions):

        # Load in data
        trials = load_trials(subject, ses)
        obj_df = load_objects(subject, ses)
        lick_times = np.load(join(data_path, 'Subjects', subject, ses, 'lick.times.npy'))
        if lick_times.shape[0] < 100:
            continue
        
        # Get timestamps of entry of goal, no-goal and control object sets
        peri_multiple_events_time_histogram(
            lick_times, np.ones(lick_times.shape[0]),
            obj_df.loc[obj_df['object'] == 2, 'times'].values,
            obj_df.loc[obj_df['object'] == 2, 'goal'].values,
            [1], t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, smoothing=0, ax=axs[j],
            pethline_kwargs=[{'color': colors['no-goal'], 'lw': 1},
                             {'color': colors['goal'], 'lw': 1}],
            errbar_kwargs=[{'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0},
                           {'color': colors['goal'], 'alpha': 0.3, 'lw': 0}],
            raster_kwargs=[{'color': colors['no-goal'], 'lw': 0.5},
                           {'color': colors['goal'], 'lw': 0.5}],
            eventline_kwargs={'lw': 0}, include_raster=True)

        axs[j].set(ylabel='Licks/s', xticks=np.arange(-T_BEFORE, T_AFTER+1),
                   yticks=[0, np.ceil(axs[j].get_ylim()[1])],
                   title=f'{ses} ({trials.shape[0]} trials)', xlabel='')
        axs[j].yaxis.set_label_coords(-0.1, 0.75)
        max_y[j] = axs[j].get_ylim()[1]

    # Place the dotted line now we know the y lim extend
    for j, ses in enumerate(sessions):
        axs[j].plot([0, 0], [0, np.ceil(np.max(max_y))], color='grey', ls='--', lw=0.75, zorder=0)

    f.suptitle(f'{subject} Object 2')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)
    if int(np.ceil(len(sessions)/4)) > 1:
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, hspace=0.4)
    else:
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.8, hspace=0.4)

    # plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], f'{subject}_task_licks_obj2.jpg'), dpi=600)

    # Create speed figure
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4, figsize=(7, 2*np.ceil(len(sessions) / 4)),
                          dpi=dpi, sharey=True)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    max_y = np.empty(len(sessions))
    for j, ses in enumerate(sessions):

        # Load in data
        trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'))
        obj_df = load_objects(subject, ses)
        wheel_times = np.load(join(data_path, 'Subjects', subject, ses, 'continuous.times.npy'))
        wheel_speed = np.load(join(data_path, 'Subjects', subject,
                              ses, 'continuous.wheelSpeed.npy'))

        # Downsample wheel speed and convert to cm/s
        wheel_speed = wheel_speed[::50] / 10
        wheel_times = wheel_times[::50]

        # Get timestamps of entry of goal, no-goal and control object sets       
        peri_event_trace(wheel_speed, wheel_times,
                         obj_df.loc[obj_df['object'] == 1, 'times'].values,
                         obj_df.loc[obj_df['object'] == 1, 'goal'].values,
                         color_palette=[colors['no-goal'], colors['goal']],
                         t_before=T_BEFORE, t_after=T_AFTER, ax=axs[j], kwargs={'zorder': 1})            
        peri_event_trace(wheel_speed, wheel_times,
                         obj_df.loc[obj_df['object'] == 2, 'times'].values,
                         obj_df.loc[obj_df['object'] == 2, 'goal'].values,
                         color_palette=[colors['no-goal'], colors['goal']],
                         t_before=T_BEFORE, t_after=T_AFTER, ax=axs[j], kwargs={'ls':'--', 'zorder': 1})
            
        axs[j].set(ylabel='Speed (cm/s)', xticks=np.arange(-T_BEFORE, T_AFTER+1),
                   title=f'{ses} ({trials.shape[0]} trials)', xlabel='')
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
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.8, hspace=0.4)

    plt.savefig(join(path_dict['fig_path'], f'{subject}_task_speed.jpg'), dpi=600)
