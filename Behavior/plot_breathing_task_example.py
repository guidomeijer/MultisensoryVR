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
from msvr_functions import load_subjects, paths, figure_style


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
subject = '459601'
ses = '20240328'

# Get subjects
subjects = load_subjects()
i = subjects.loc[subjects['SubjectID'] == subject].index[0]

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

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

# %% Plot
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=breathing_df, x='time', y='psd', hue='object', err_kws={'lw': 0},
             errorbar='se', ax=ax1, hue_order=['Goal', 'Distractor', 'Control'],
             palette=[colors['goal'], colors['no-goal'], colors['control']])

ax1.set(ylabel='Breathing amplitude (PSD)', xticks=np.arange(T_PLOT[0], T_PLOT[1]+1),
           xlim=T_PLOT, yticks=np.arange(0, 21, 5), xlabel='')
ax1.get_legend().remove()
ax1.plot([0, 0], ax1.get_ylim(), color='k', ls='--', lw=0.75)
ax1.set(ylabel='Amplitude (PSD)', xlabel='Time from object entry (s)',
        title='Breathing')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(path_dict['fig_path'], f'{subject}_task_breathing_example.jpg'), dpi=600)
plt.savefig(join(path_dict['fig_path'], f'{subject}_task_breathing_example.pdf'))
