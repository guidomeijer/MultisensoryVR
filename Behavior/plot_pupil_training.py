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
import datetime
from os.path import join, isfile
import pandas as pd
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style,
                            peri_multiple_events_time_histogram)

# Settings
T_BEFORE = 2
T_AFTER = 4
BIN_SIZE = 0.2
SMOOTHING = 0.1
PLOT_SUBJECTS = ['452505', '452506']

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

# Loop over subjects
for i, subject in enumerate(PLOT_SUBJECTS):

    # List sessions
    sessions = os.listdir(join(data_path, 'Subjects', subject))

    # Only include sessions with pupil and trial data 
    incl_ses = np.zeros(len(sessions)).astype(int)
    for j, ses in enumerate(sessions):
        if ((isfile(join(data_path, 'Subjects', subject, ses, 'pupil.csv'))) &
            (isfile(join(data_path, 'Subjects', subject, ses, 'trials.csv')))):
            incl_ses[j] = 1
    sessions = np.array(sessions)[incl_ses.astype(bool)]
     
    # Select training sessions
    ses_date = [datetime.datetime.strptime(i, '%Y%m%d').date() for i in sessions]
    ses_date = [k for k in ses_date if k < subjects.loc[subjects['SubjectID'] == subject,
                                                        'DateFinalTask'].values[0]]
    ses_date = ses_date[-12:]  # only plot last 12 sessions
    sessions = [datetime.datetime.strftime(i, '%Y%m%d') for i in ses_date]

    # Create figure
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4, figsize=(7,  2*np.ceil(len(sessions) / 4)),
                          dpi=dpi)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    min_y, max_y = np.empty(len(sessions)), np.empty(len(sessions))
    for j, ses in enumerate(sessions):

        # Load in data
        trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'))
        pupil_df = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'pupil.csv'))
        timestamps = np.load(join(data_path, 'Subjects', subject, ses, 'camera.times.npy'))
        
        print(pupil_df.shape[0])
        print(timestamps.shape[0])
        
        # Drop last frames
        pupil_size = pupil_df['width_smooth'].values
        timestamps = timestamps[:pupil_size.shape[0]]
        
        # Plot
        all_obj_enters = np.concatenate(
            (trials['enterObj1'], trials['enterObj2'], trials['enterObj3']))
        all_obj_ids = np.concatenate(
            (np.ones(trials['enterObj1'].shape),
             np.ones(trials['enterObj2'].shape)*2,
             np.ones(trials['enterObj3'].shape)*3))
        if j == 0:
            peri_event_trace(pupil_size, timestamps, all_obj_enters,
                             event_ids=all_obj_ids, color_palette='Set2',
                             event_labels=['1', '2', '3'],
                             t_before=T_BEFORE, t_after=T_AFTER, ax=axs[j], kwargs={'zorder': 1})
        else:
            peri_event_trace(pupil_size, timestamps, all_obj_enters,
                             event_ids=all_obj_ids, color_palette='Set2',
                             t_before=T_BEFORE, t_after=T_AFTER, ax=axs[j], kwargs={'zorder': 1})
        axs[j].set(xticks=np.arange(-2, 7, 2), title=f'{ses}', xlabel='')
        axs[j].set_ylabel('Pupil size (px)', labelpad=-10)
        min_y[j], max_y[j] = axs[j].get_ylim()[0], axs[j].get_ylim()[1]

    # Place the dotted line now we know the y lim extend
    for j, ses in enumerate(sessions):
        axs[j].plot([0, 0], [np.floor(np.min(min_y)), np.ceil(np.max(max_y))],
                    color='grey', ls='--', lw=0.75, zorder=0)
        axs[j].set(ylim=[np.min(min_y), np.max(max_y)],
                   yticks=[np.round(np.min(min_y)), np.round(np.ceil(np.max(max_y)))])

    f.suptitle(f'{subjects.loc[subjects["SubjectID"] == subject, "Nickname"].values[0]} ({subject})')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)
    if int(np.ceil(len(sessions)/4)) > 1:
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, hspace=0.4)
    else:
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.4)
    plt.savefig(join(path_dict['fig_path'], f'{subject}_pupil_objects.jpg'), dpi=600)
    
    # Create figure reward times
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4, figsize=(7,  2*np.ceil(len(sessions) / 4)),
                          dpi=dpi)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    min_y, max_y = np.empty(len(sessions)), np.empty(len(sessions))
    for j, ses in enumerate(sessions):

        # Load in data
        reward_times = np.load(join(data_path, 'Subjects', subject, ses, 'reward.times.npy'))
        pupil_df = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'pupil.csv'))
        timestamps = np.load(join(data_path, 'Subjects', subject, ses, 'camera.times.npy'))
                
        # Drop last frames
        pupil_size = pupil_df['width_smooth'].values
        timestamps = timestamps[:pupil_size.shape[0]]
        
        # Plot
        peri_event_trace(pupil_size, timestamps, reward_times,
                         event_ids=np.ones(reward_times.shape), color_palette='Reds',
                         t_before=T_BEFORE, t_after=T_AFTER, ax=axs[j], kwargs={'zorder': 1})
        axs[j].set(xticks=np.arange(-2, 7, 2), title=f'{ses}', xlabel='')
        axs[j].set_ylabel('Pupil size (px)', labelpad=-10)
        min_y[j], max_y[j] = axs[j].get_ylim()[0], axs[j].get_ylim()[1]

    # Place the dotted line now we know the y lim extend
    for j, ses in enumerate(sessions):
        axs[j].plot([0, 0], [np.floor(np.min(min_y)), np.ceil(np.max(max_y))],
                    color='grey', ls='--', lw=0.75, zorder=0)
        axs[j].set(ylim=[np.min(min_y), np.max(max_y)],
                   yticks=[np.round(np.min(min_y)), np.round(np.ceil(np.max(max_y)))])

    f.suptitle(f'{subjects.loc[subjects["SubjectID"] == subject, "Nickname"].values[0]} ({subject})')
    f.text(0.5, 0.04, 'Time from reward (s)', ha='center')
    sns.despine(trim=True)
    if int(np.ceil(len(sessions)/4)) > 1:
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, hspace=0.4)
    else:
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.4)
    plt.savefig(join(path_dict['fig_path'], f'{subject}_pupil_reward.jpg'), dpi=600)
    
    """
    f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
    ax1.plot(timestamps[2200:2800], pupil_size[2200:2800])
    ax1.set(ylabel='Pupil size (px)', xlabel='Time (s)')
    """