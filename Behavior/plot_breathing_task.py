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
from scipy.signal import spectrogram, butter, filtfilt
from os.path import join, isfile
import pandas as pd
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style,
                            peri_multiple_events_time_histogram)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


# Settings
T_BEFORE = 3.5
T_AFTER = 4.5
T_PLOT = [-2, 3]
BIN_SIZE = 0.2
SMOOTHING = 0.1
WIN_SIZE = 2  # s
WIN_SHIFT = 0.05  # s
FS = 1000  # sampling rate
FREQ = [5, 10]

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

    # Discard sessions that don't have breathing
    no_breathing_ses = np.zeros(len(sessions))
    for j, ses in enumerate(sessions):
        if not isfile(join(data_path, 'Subjects', subject, ses, 'continuous.breathing.npy')):
            no_breathing_ses[j] = 1
    sessions = np.array(sessions)[~no_breathing_ses.astype(int).astype(bool)]

    # Select final task sessions
    ses_date = [datetime.datetime.strptime(i, '%Y%m%d').date() for i in sessions]
    ses_date = [k for k in ses_date if k >= subjects.iloc[i, 3]]
    sessions = [datetime.datetime.strftime(i, '%Y%m%d') for i in ses_date]
    if len(sessions) == 0:
        continue

    # Create breathing figure
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4, figsize=(7,  2*np.ceil(len(sessions) / 4)),
                          dpi=dpi)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    max_y = np.empty(len(sessions))
    for j, ses in enumerate(sessions):

        # Load in data
        trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'))
        timestamps = np.load(join(data_path, 'Subjects', subject, ses, 'continuous.times.npy'))
        breathing = np.load(join(data_path, 'Subjects', subject, ses, 'continuous.breathing.npy'))

        # Get timestamps of entry of goal, no-goal and control object sets
        goal_obj_enters = np.concatenate((
            trials.loc[trials['soundId'] == 1, f'enterObj{subjects.loc[i, "Sound1Obj"]}'],
            trials.loc[trials['soundId'] == 2, f'enterObj{subjects.loc[i, "Sound2Obj"]}']))
        nogoal_obj_enters = np.concatenate((
            trials.loc[trials['soundId'] == 1, f'enterObj{subjects.loc[i, "Sound2Obj"]}'],
            trials.loc[trials['soundId'] == 2, f'enterObj{subjects.loc[i, "Sound1Obj"]}']))
        control_obj_enters = trials[f'enterObj{subjects.loc[i, "ControlObject"]}'].values
        goal_obj_enters = np.sort(goal_obj_enters[~np.isnan(goal_obj_enters)])
        nogoal_obj_enters = np.sort(nogoal_obj_enters[~np.isnan(nogoal_obj_enters)])
        control_obj_enters = np.sort(control_obj_enters[~np.isnan(control_obj_enters)])

        # Filter breathing signal
        breathing_filt = butter_bandpass_filter(breathing, 2, 50, 1000, order=1)

        # Compute breathing spectogram per trial
        breathing_df = pd.DataFrame()
        all_spec = []
        for k, this_onset in enumerate(goal_obj_enters):
            this_ind = (timestamps >= this_onset - T_BEFORE) & (timestamps <= this_onset + T_AFTER)
            freq, time, this_spec = spectrogram(breathing_filt[this_ind], fs=FS,
                                                nperseg=WIN_SIZE*FS,
                                                noverlap=(WIN_SIZE*FS)-(WIN_SHIFT*FS))
            all_spec.append(this_spec)
        all_t_ax = np.array([i.shape[1] for i in all_spec])
        all_spec = [i for (i, v) in zip(all_spec, all_t_ax == time.shape[0]) if v]
        all_spec = np.dstack(all_spec)
        no_goal = np.mean(all_spec[(freq >= FREQ[0]) & (freq <= FREQ[1]), :, :], axis=0)
        time_ax = time - T_BEFORE
        this_df = pd.melt(pd.DataFrame(columns=time_ax, data=no_goal.T),
                          value_name='psd', var_name='time')
        this_df['object'] = 'Goal'
        breathing_df = pd.concat((breathing_df, this_df))

        # No goal
        all_spec = []
        for k, this_onset in enumerate(nogoal_obj_enters):
            this_ind = (timestamps >= this_onset - T_BEFORE) & (timestamps <= this_onset + T_AFTER)
            freq, time, this_spec = spectrogram(breathing_filt[this_ind], fs=FS,
                                                nperseg=WIN_SIZE*FS,
                                                noverlap=(WIN_SIZE*FS)-(WIN_SHIFT*FS))
            all_spec.append(this_spec)
        all_t_ax = np.array([i.shape[1] for i in all_spec])
        all_spec = [i for (i, v) in zip(all_spec, all_t_ax == time.shape[0]) if v]
        all_spec = np.dstack(all_spec)
        no_goal = np.mean(all_spec[(freq >= FREQ[0]) & (freq <= FREQ[1]), :, :], axis=0)
        time_ax = time - T_BEFORE
        this_df = pd.melt(pd.DataFrame(columns=time_ax, data=no_goal.T),
                          value_name='psd', var_name='time')
        this_df['object'] = 'Distractor'
        breathing_df = pd.concat((breathing_df, this_df))

        # Control
        all_spec = []
        for k, this_onset in enumerate(control_obj_enters):
            this_ind = (timestamps >= this_onset - T_BEFORE) & (timestamps <= this_onset + T_AFTER)
            freq, time, this_spec = spectrogram(breathing_filt[this_ind], fs=FS,
                                                nperseg=WIN_SIZE*FS,
                                                noverlap=(WIN_SIZE*FS)-(WIN_SHIFT*FS))
            all_spec.append(this_spec)
        all_t_ax = np.array([i.shape[1] for i in all_spec])
        all_spec = [i for (i, v) in zip(all_spec, all_t_ax == time.shape[0]) if v]
        all_spec = np.dstack(all_spec)
        no_goal = np.mean(all_spec[(freq >= FREQ[0]) & (freq <= FREQ[1]), :, :], axis=0)
        time_ax = time - T_BEFORE
        this_df = pd.melt(pd.DataFrame(columns=time_ax, data=no_goal.T),
                          value_name='psd', var_name='time')
        this_df['object'] = 'Control'
        breathing_df = pd.concat((breathing_df, this_df))

        # Plot
        sns.lineplot(data=breathing_df, x='time', y='psd', hue='object', err_kws={'lw': 0},
                     errorbar='se', ax=axs[j], hue_order=['Goal', 'Distractor', 'Control'],
                     palette=[colors['goal'], colors['no-goal'], colors['control']])

        if j == 0:
            axs[j].set(ylabel='Breathing amplitude (PSD)', xticks=np.arange(T_PLOT[0], T_PLOT[1]+1),
                       xlim=T_PLOT, yticks=np.arange(2, 11, 2), xlabel='', title=ses)
            g = axs[j].legend(title='', prop={'size': 5.5})
        else:
            axs[j].set(xticks=np.arange(T_PLOT[0], T_PLOT[1]+1), xlim=T_PLOT, xlabel='', yticks=[],
                       title=ses)
            axs[j].get_yaxis().set_visible(False)
            axs[j].get_legend().remove()
        axs[j].plot([0, 0], axs[j].get_ylim(), color='k', ls='--', lw=0.75)

    f.suptitle(f'{subjects.iloc[i, 1]} ({subject})')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)

    if int(np.ceil(len(sessions)/4)) > 1:
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.9, hspace=0.4)
    else:
        plt.subplots_adjust(left=0.08, bottom=0.2, right=0.95, top=0.8, hspace=0.4)

    # plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], f'{subject}_task_breathing.jpg'), dpi=600)
