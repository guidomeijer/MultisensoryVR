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
T_BEFORE = 4
T_AFTER = 8
BIN_SIZE = 0.2
SMOOTHING = 0.1
WIN_SIZE = 2  # s
WIN_SHIFT = 0.05  # s
FS = 1000  # sampling rate

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

    """
    # Select final task sessions
    ses_date = [datetime.datetime.strptime(i, '%Y%m%d').date() for i in sessions]
    ses_date = [k for k in ses_date if k >= subjects.iloc[i, 3]]
    sessions = [datetime.datetime.strftime(i, '%Y%m%d') for i in ses_date]
    if len(sessions) == 0:
        continue
    """

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
        all_obj_enters = np.concatenate((goal_obj_enters, nogoal_obj_enters, control_obj_enters))
        all_obj_ids = np.concatenate(
            (np.ones(goal_obj_enters.shape[0]),
             np.ones(nogoal_obj_enters.shape[0])*2,
             np.ones(control_obj_enters.shape[0])*3))
        all_obj_ids = all_obj_ids[~np.isnan(all_obj_enters)]
        all_obj_enters = all_obj_enters[~np.isnan(all_obj_enters)]

        # Filter breathing signal
        breathing_filt = butter_bandpass_filter(breathing, 2, 50, 1000, order=1)

        # Compute breathing spectogram per trial
        all_spec = []
        for k, this_onset in enumerate(all_obj_enters):
            this_ind = (timestamps >= this_onset - T_BEFORE) & (timestamps <= this_onset + T_AFTER)
            freq, time, this_spec = spectrogram(breathing_filt[this_ind], fs=FS,
                                                nperseg=WIN_SIZE*FS,
                                                noverlap=(WIN_SIZE*FS)-(WIN_SHIFT*FS))
            all_spec.append(this_spec[freq <= 15, :])
            freq = freq[freq <= 15]
        all_t_ax = np.array([i.shape[1] for i in all_spec])
        all_spec = [i for (i, v) in zip(all_spec, all_t_ax == time.shape[0]) if v]
        all_spec = np.dstack(all_spec)
        time_ax = time - T_BEFORE

        # Plot
        axs[j].imshow(np.mean(all_spec, axis=2), aspect='auto',
                      extent=[-T_BEFORE, T_AFTER, freq[-1], freq[0]])
        axs[j].plot([0, 0], axs[j].get_ylim(), color='w', ls='--')
        axs[j].invert_yaxis()
        axs[j].set(ylabel='Frequency (Hz)', xticks=np.arange(-T_BEFORE, T_AFTER+1, 2),
                   yticks=np.arange(0, 16, 5))

    f.suptitle(f'{subjects.iloc[i, 1]} ({subject})')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)
    if int(np.ceil(len(sessions)/4)) > 1:
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.9, hspace=0.4)
    else:
        plt.subplots_adjust(left=0.08, bottom=0.2, right=0.95, top=0.8, hspace=0.4)

    # plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], f'{subject}_task_breathing.jpg'), dpi=600)
