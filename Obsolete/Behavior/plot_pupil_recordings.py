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
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from scipy.signal import spectrogram, butter, filtfilt, hilbert
from msvr_functions import (load_subjects, paths, figure_style, event_aligned_averages,
                            load_objects)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y

# Settings
T_BEFORE = 2
T_AFTER = 3
BIN_SIZE = 0.2
FREQ = [5, 10]
BASELINE = [-2, -1]

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

# Get time
bin_edges = np.round(np.arange(-T_BEFORE, T_AFTER + (BIN_SIZE/2), step=BIN_SIZE), 1)
time_ax = bin_edges[:-1] + (BIN_SIZE/2)

# Load in recording sessions
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

obj1_df, obj2_df = pd.DataFrame(), pd.DataFrame()
for i, subject in enumerate(np.unique(rec['subject'])):
    print(f'Processing subject {subject}..')
    
    # Take slice of dataframe
    sub_rec = rec[rec['subject'] == subject]

    for j, date in enumerate(np.unique(sub_rec['date'])):
           
        if not (data_path / 'Subjects' / subject / date / 'pupil.csv').is_file():
            print(f'No pupil data for {subject} {date}')
            continue
        
        # Load in data
        trials_df = pd.read_csv(data_path / 'Subjects' / subject / date / 'trials.csv')
        pupil_df = pd.read_csv(data_path / 'Subjects' / subject / date / 'pupil.csv')
        timestamps = np.load(data_path / 'Subjects' / subject / date / 'camera.times.npy')
        all_obj_df = load_objects(subject, date)
    
        # Correct for timestamps difference
        timestamps = timestamps[:pupil_df.shape[0]]
    
        # Transform into percentage
        perc_pupil = ((pupil_df['width_smooth'] - np.percentile(pupil_df['width_smooth'], 0.05))
                      / np.percentile(pupil_df['width_smooth'], 0.05)) * 100
        
        # Do it per object
        obj1_goal_df = event_aligned_averages(
            perc_pupil, timestamps,
            all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['object'] == 1), 'times'].values,
            bin_edges, return_df=True, baseline=BASELINE)
        obj1_no_goal_df = event_aligned_averages(
            perc_pupil, timestamps,
            all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] == 1), 'times'].values,
            bin_edges, return_df=True, baseline=BASELINE)
        obj1_goal_df['goal'] = 1
        obj1_goal_df['subject'] = subject
        obj1_no_goal_df['goal'] = 0
        obj1_no_goal_df['subject'] = subject
        obj1_df = pd.concat([obj1_df, obj1_goal_df, obj1_no_goal_df], ignore_index=True)
        
        obj2_goal_df = event_aligned_averages(
            perc_pupil, timestamps,
            all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['object'] == 2), 'times'].values,
            bin_edges, return_df=True, baseline=BASELINE)
        obj2_no_goal_df = event_aligned_averages(
            perc_pupil, timestamps,
            all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] == 2), 'times'].values,
            bin_edges, return_df=True, baseline=BASELINE)
        obj2_goal_df['goal'] = 1
        obj2_goal_df['subject'] = subject
        obj2_no_goal_df['goal'] = 0
        obj2_no_goal_df['subject'] = subject
        obj2_df = pd.concat([obj2_df, obj2_goal_df, obj2_no_goal_df], ignore_index=True)
    
# %%
n_subjects = np.unique(obj1_df['subject']).shape[0]
f, axs = plt.subplots(1, n_subjects, figsize=(1.75*n_subjects, 2), dpi=dpi)
for (i, subject) in enumerate(np.unique(obj1_df['subject'])):
    sns.lineplot(data=obj1_df[obj1_df['subject'] == subject], x='time', y='value', hue='goal',
                 ax=axs[i], hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']],
                 errorbar='se', err_kws={'lw': 0}, legend=None)
    axs[i].set(ylabel='Pupil size (%)', xlabel='Time from object entry (s)',
               title=subject, xticks=[-2, -1, 0, 1, 2, 3])

plt.suptitle('First object')
sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(join(path_dict['google_drive_fig_path'], 'pupil_recordings_obj1.pdf'))

# %%

f, axs = plt.subplots(1, n_subjects, figsize=(1.75*n_subjects, 2), dpi=dpi)
for (i, subject) in enumerate(np.unique(obj2_df['subject'])):
    sns.lineplot(data=obj2_df[obj2_df['subject'] == subject], x='time', y='value', hue='goal',
                 ax=axs[i], hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']],
                 errorbar='se', err_kws={'lw': 0}, legend=None)
    axs[i].set(ylabel='Pupil size (%)', xlabel='Time from object entry (s)',
               title=subject, xticks=[-2, -1, 0, 1, 2, 3])

plt.suptitle('Second object')
sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(join(path_dict['google_drive_fig_path'], 'pupil_recordings_obj2.pdf'))
