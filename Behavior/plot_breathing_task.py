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
from scipy.signal import spectrogram, butter, filtfilt, hilbert
from os.path import join, isfile
import pandas as pd
from msvr_functions import load_subjects, paths, figure_style


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


# Settings
T_BEFORE = 1.5
T_AFTER = 1.5
T_PLOT = [-1, 1]
Y_LIM = [2, 16]
WIN_SIZE = 0.25  # s
WIN_SHIFT = 0.01  # s
FS = 1000  # sampling rate
FREQ = [5, 10]
TIME_AX = np.linspace(-T_BEFORE, T_AFTER, num=int((T_BEFORE + T_AFTER) * FS))
PLOT_SUBJECTS = ['462910', '462911']

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
    sessions = os.listdir(join(data_path, 'Subjects', subject))

    # Discard sessions that don't have breathing
    no_breathing_ses = np.zeros(len(sessions))
    for j, ses in enumerate(sessions):
        if not isfile(join(data_path, 'Subjects', subject, ses, 'continuous.breathing.npy')):
            no_breathing_ses[j] = 1
    sessions = np.array(sessions)[~no_breathing_ses.astype(int).astype(bool)]

    # Select final task sessions
    ses_date = np.array([datetime.datetime.strptime(i[:8], '%Y%m%d').date() for i in sessions])
    sessions = sessions[ses_date >= subjects.loc[subjects['SubjectID'] == subject,
                                                 'DateFinalTask'].values[0]]
    if len(sessions) == 0:
        continue

    # Create breathing figure
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4,
                          figsize=(7,  2*np.ceil(len(sessions) / 4)),
                          dpi=dpi)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    max_y = np.empty(len(sessions))
    for j, ses in enumerate(sessions):

        # Load in data
        trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'))
        timestamps = np.load(join(data_path, 'Subjects', subject, ses, 'continuous.times.npy'))
        breathing = np.load(join(data_path, 'Subjects', subject, ses, 'continuous.breathing.npy'))
        
        # Sometimes there is a one sample difference between timestamps and breathing
        if timestamps.shape[0] > breathing.shape[0]:
            timestamps = timestamps[:breathing.shape[0]]
        elif timestamps.shape[0] < breathing.shape[0]:
            breathing = breathing[:timestamps.shape[0]]

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
        breathing_filt = butter_bandpass_filter(breathing, FREQ[0], FREQ[1], 1000, order=1)

        # Compute breathing spectogram per trial
        sniffing_df = pd.DataFrame()
        all_amp = []
        for k, this_onset in enumerate(goal_obj_enters):
            this_ind = (timestamps >= this_onset - T_BEFORE) & (timestamps < this_onset + T_AFTER)
            if np.sum(this_ind) == TIME_AX.shape[0] + 1:
                this_ind = (timestamps > this_onset - T_BEFORE) & (timestamps < this_onset + T_AFTER)
            elif np.sum(this_ind) == TIME_AX.shape[0] - 1:
                this_ind = (timestamps >= this_onset - T_BEFORE) & (timestamps <= this_onset + T_AFTER)
            elif np.sum(this_ind) != TIME_AX.shape[0]:
                continue
            analytic_signal = hilbert(breathing_filt[this_ind])
            amplitude_envelope = np.abs(analytic_signal)
            all_amp.append(amplitude_envelope)
        this_df = pd.melt(pd.DataFrame(columns=TIME_AX, data=np.vstack(all_amp)),
                          value_name='freq', var_name='time')
        this_df['object'] = 'Goal'
        sniffing_df = pd.concat((sniffing_df, this_df))
        
        # No goal
        all_amp = []
        for k, this_onset in enumerate(nogoal_obj_enters):
            this_ind = (timestamps >= this_onset - T_BEFORE) & (timestamps < this_onset + T_AFTER)
            if np.sum(this_ind) == TIME_AX.shape[0] + 1:
                this_ind = (timestamps > this_onset - T_BEFORE) & (timestamps < this_onset + T_AFTER)
            elif np.sum(this_ind) == TIME_AX.shape[0] - 1:
                this_ind = (timestamps >= this_onset - T_BEFORE) & (timestamps <= this_onset + T_AFTER)
            elif np.sum(this_ind) != TIME_AX.shape[0]:
                continue
            analytic_signal = hilbert(breathing_filt[this_ind])
            amplitude_envelope = np.abs(analytic_signal)
            all_amp.append(amplitude_envelope)
        this_df = pd.melt(pd.DataFrame(columns=TIME_AX, data=np.vstack(all_amp)),
                          value_name='freq', var_name='time')
        this_df['object'] = 'Distractor'
        sniffing_df = pd.concat((sniffing_df, this_df))

        # Control
        all_amp = []
        for k, this_onset in enumerate(control_obj_enters):
            this_ind = (timestamps >= this_onset - T_BEFORE) & (timestamps < this_onset + T_AFTER)
            if np.sum(this_ind) == TIME_AX.shape[0] + 1:
                this_ind = (timestamps > this_onset - T_BEFORE) & (timestamps < this_onset + T_AFTER)
            elif np.sum(this_ind) == TIME_AX.shape[0] - 1:
                this_ind = (timestamps >= this_onset - T_BEFORE) & (timestamps <= this_onset + T_AFTER)
            elif np.sum(this_ind) != TIME_AX.shape[0]:
                continue
            analytic_signal = hilbert(breathing_filt[this_ind])
            amplitude_envelope = np.abs(analytic_signal)
            all_amp.append(amplitude_envelope)
        this_df = pd.melt(pd.DataFrame(columns=TIME_AX, data=np.vstack(all_amp)),
                          value_name='freq', var_name='time')
        this_df['object'] = 'Control'
        sniffing_df = pd.concat((sniffing_df, this_df))

        # Plot sniffing power
        sns.lineplot(data=sniffing_df, x='time', y='freq', hue='object', err_kws={'lw': 0},
                     errorbar='se', ax=axs[j], hue_order=['Goal', 'Distractor', 'Control'],
                     palette=[colors['goal'], colors['no-goal'], colors['control']])

        if np.mod(j, 4) == 0:
            axs[j].set(ylabel='Sniffing amplitude (mV)', xticks=np.arange(T_PLOT[0], T_PLOT[1]+1, 0.25),
                       xlim=T_PLOT, yticks=np.arange(Y_LIM[0], Y_LIM[1]+1, 2), xlabel='',
                       title=ses, ylim=Y_LIM)
            g = axs[j].legend(title='', prop={'size': 5.5})
        else:
            axs[j].set(xticks=np.arange(T_PLOT[0], T_PLOT[1]+1, 0.5), xlim=T_PLOT, xlabel='', yticks=[],
                       title=ses, ylim=Y_LIM)
            axs[j].get_yaxis().set_visible(False)
            axs[j].get_legend().remove()
        axs[j].plot([0, 0], axs[j].get_ylim(), color='k', ls='--', lw=0.75)
        
    f.suptitle(f'{subject}')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)

    if int(np.ceil(len(sessions)/4)) > 1:
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.9, hspace=0.4)
    else:
        plt.subplots_adjust(left=0.08, bottom=0.2, right=0.95, top=0.8, hspace=0.4)

    # plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], f'{subject}_task_breathing.jpg'), dpi=600)
