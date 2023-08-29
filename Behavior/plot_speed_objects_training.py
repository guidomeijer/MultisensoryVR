# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:57:35 2023

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
import os
from os.path import join, isfile
import pandas as pd
from msvr_functions import load_subjects, paths, peri_event_trace, figure_style

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

# Loop over subjects
for i, subject in enumerate(subjects['SubjectID']):

    # List sessions
    sessions = os.listdir(join(data_path, 'Subjects', subject))

    # Create figure for all object entry
    f, axs = plt.subplots(int(np.ceil(len(sessions)/5)), 5, figsize=(7, 2*np.ceil(len(sessions) / 5)),
                          dpi=dpi, sharey=True, sharex=True)
    if len(sessions) > 5:
        axs = np.concatenate(axs)

    for j, ses in enumerate(sessions):
        if isfile(join(path_dict['local_data_path'], 'Subjects', subject, ses, 'trials.csv')):

            # Load in data
            trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'),
                                 index_col=0)
            wheel_times = np.load(join(data_path, 'Subjects', subject, ses, 'wheel.times.npy'))
            wheel_speed = np.load(join(data_path, 'Subjects', subject, ses, 'wheel.speed.npy'))

            # Downsample wheel speed and convert to cm/s
            wheel_speed = wheel_speed[::50] / 10
            wheel_times = wheel_times[::50]

            # Plot
            all_obj_enters = np.concatenate(
                (trials['enterObj1'], trials['enterObj2'], trials['enterObj3']))
            #all_obj_enters = np.sort(all_obj_enters[~np.isnan(all_obj_enters)])
            peri_event_trace(wheel_speed, wheel_times, all_obj_enters,
                             event_ids=np.ones(all_obj_enters.shape),
                             t_before=2, t_after=6, ax=axs[j], kwargs={'zorder': 1})
            axs[j].set(ylabel='Speed (cm/s)', xticks=np.arange(-2, 7, 2), ylim=[0, 4],
                       title=f'{ses}', xlabel='')
            axs[j].plot([0, 0], axs[j].get_ylim(), color='grey', ls='--', lw=0.75, zorder=0)

    f.suptitle(f'{subjects.iloc[i, 1]} ({subject})')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)
    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.8)
    plt.savefig(join(path_dict['fig_path'], f'{subject}_all_objects_entry.jpg'), dpi=600)

    # Create figure for object entry
    f, axs = plt.subplots(int(np.ceil(len(sessions)/5)), 5, figsize=(7, 2*np.ceil(len(sessions) / 5)),
                          dpi=dpi, sharey=True, sharex=True)
    if len(sessions) > 5:
        axs = np.concatenate(axs)

    for j, ses in enumerate(sessions):
        if isfile(join(path_dict['local_data_path'], 'Subjects', subject, ses, 'trials.csv')):

            # Load in data
            trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'),
                                 index_col=0)
            wheel_times = np.load(join(data_path, 'Subjects', subject, ses, 'wheel.times.npy'))
            wheel_speed = np.load(join(data_path, 'Subjects', subject, ses, 'wheel.speed.npy'))

            # Downsample wheel speed and convert to cm/s
            wheel_speed = wheel_speed[::50] / 10
            wheel_times = wheel_times[::50]

            # Plot
            all_obj_enters = np.concatenate(
                (trials['enterObj1'], trials['enterObj2'], trials['enterObj3']))
            all_obj_ids = np.concatenate(
                (np.ones(trials['enterObj1'].shape),
                 np.ones(trials['enterObj2'].shape)*2,
                 np.ones(trials['enterObj3'].shape)*3))
            if j+1 == len(sessions):
                peri_event_trace(wheel_speed, wheel_times, all_obj_enters,
                                 event_ids=all_obj_ids, color_palette='Set2',
                                 event_labels=['1', '2', '3'],
                                 t_before=2, t_after=6, ax=axs[j], kwargs={'zorder': 1})
            else:
                peri_event_trace(wheel_speed, wheel_times, all_obj_enters,
                                 event_ids=all_obj_ids, color_palette='Set2',
                                 t_before=2, t_after=6, ax=axs[j], kwargs={'zorder': 1})
            axs[j].set(ylabel='Speed (cm/s)', xticks=np.arange(-2, 7, 2), ylim=[0, 4],
                       title=f'{ses}', xlabel='')
            axs[j].plot([0, 0], axs[j].get_ylim(), color='grey', ls='--', lw=0.75, zorder=0)

    f.suptitle(f'{subjects.iloc[i, 1]} ({subject})')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)
    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.8)
    plt.savefig(join(path_dict['fig_path'], f'{subject}_object_entry.jpg'), dpi=600)

    # Create figure for rewards
    f, axs = plt.subplots(int(np.ceil(len(sessions)/5)), 5, figsize=(7, 2*np.ceil(len(sessions) / 5)),
                          dpi=dpi, sharey=True, sharex=True)
    if len(sessions) > 5:
        axs = np.concatenate(axs)

    for j, ses in enumerate(sessions):
        if isfile(join(data_path, 'Subjects', subject, ses, 'trials.csv')):

            # Load in data
            reward_times = np.load(join(data_path, 'Subjects', subject, ses, 'reward.times.npy'))
            wheel_times = np.load(join(data_path, 'Subjects', subject, ses, 'wheel.times.npy'))
            wheel_speed = np.load(join(data_path, 'Subjects', subject, ses, 'wheel.speed.npy'))

            # Downsample wheel speed and convert to cm/s
            wheel_speed = wheel_speed[::50] / 10
            wheel_times = wheel_times[::50]

            # Plot
            peri_event_trace(wheel_speed, wheel_times, reward_times,
                             event_ids=np.ones(reward_times.shape),
                             t_before=2, t_after=6, ax=axs[j], kwargs={'zorder': 1})
            axs[j].set(ylabel='Speed (cm/s)', xticks=np.arange(-2, 7, 2), ylim=[0, 4],
                       title=f'{ses}', xlabel='')
            axs[j].plot([0, 0], axs[j].get_ylim(), color='grey', ls='--', lw=0.75, zorder=0)

    f.suptitle(f'{subjects.iloc[i, 1]} ({subject})')
    f.text(0.5, 0.04, 'Time from reward delivery (s)', ha='center')
    sns.despine(trim=True)
    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.8)
    plt.savefig(join(path_dict['fig_path'], f'{subject}_reward_delivery.jpg'), dpi=600)
