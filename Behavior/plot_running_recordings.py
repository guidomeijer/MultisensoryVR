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
from msvr_functions import (load_subjects, paths, figure_style, event_aligned_averages, load_objects)

# Settings
T_BEFORE = 2
T_AFTER = 3
BIN_SIZE = 0.2

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

both_obj_df, obj1_df, obj2_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for i, subject in enumerate(np.unique(rec['subject'])):
    print(f'Processing subject {subject}..')
    
    # Take slice of dataframe
    sub_rec = rec[rec['subject'] == subject]

    for j, date in enumerate(sub_rec['date']):
    
        # Get reward contingencies
        sound1_obj = subjects.loc[subjects['SubjectID'] == subject, "Sound1Obj"].values[0]
        sound2_obj = subjects.loc[subjects['SubjectID'] == subject, "Sound2Obj"].values[0]
        control_obj = subjects.loc[subjects['SubjectID'] == subject, "ControlObject"].values[0]    
       
        # Load in data
        trials = pd.read_csv(join(data_path, 'Subjects', subject, date, 'trials.csv'))
        wheel_times = np.load(join(data_path, 'Subjects', subject, date, 'continuous.times.npy'))
        wheel_speed = np.load(join(data_path, 'Subjects', subject, date, 'continuous.wheelSpeed.npy'))
        all_obj_df = load_objects(subject, date)
    
        # Convert to cm/s and downsample
        wheel_speed = wheel_speed[::50] / 10
        wheel_times = wheel_times[::50]
        
        # Get average goal and no goal speed for both objects
        this_goal_df = event_aligned_averages(
            wheel_speed, wheel_times,
            all_obj_df.loc[all_obj_df['goal'] == 1, 'times'].values,
            bin_edges, return_df=True)
        this_no_goal_df = event_aligned_averages(
            wheel_speed, wheel_times,
            all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] != 3), 'times'].values,
            bin_edges, return_df=True)
        
        # Add to subject dataframe
        this_goal_df['goal'] = 1
        this_goal_df['subject'] = subject
        this_no_goal_df['goal'] = 0
        this_no_goal_df['subject'] = subject
        both_obj_df = pd.concat([both_obj_df, this_goal_df, this_no_goal_df], ignore_index=True)
        
        # Do it per object
        obj1_goal_df = event_aligned_averages(
            wheel_speed, wheel_times,
            all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['object'] == 1), 'times'].values,
            bin_edges, return_df=True)
        obj1_no_goal_df = event_aligned_averages(
            wheel_speed, wheel_times,
            all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] == 1), 'times'].values,
            bin_edges, return_df=True)
        obj1_goal_df['goal'] = 1
        obj1_goal_df['subject'] = subject
        obj1_no_goal_df['goal'] = 0
        obj1_no_goal_df['subject'] = subject
        obj1_df = pd.concat([obj1_df, obj1_goal_df, obj1_no_goal_df], ignore_index=True)
        
        obj2_goal_df = event_aligned_averages(
            wheel_speed, wheel_times,
            all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['object'] == 2), 'times'].values,
            bin_edges, return_df=True)
        obj2_no_goal_df = event_aligned_averages(
            wheel_speed, wheel_times,
            all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] == 2), 'times'].values,
            bin_edges, return_df=True)
        obj2_goal_df['goal'] = 1
        obj2_goal_df['subject'] = subject
        obj2_no_goal_df['goal'] = 0
        obj2_no_goal_df['subject'] = subject
        obj2_df = pd.concat([obj2_df, obj2_goal_df, obj2_no_goal_df], ignore_index=True)
    
   
# %% Plot

n_subjects = np.unique(both_obj_df['subject']).shape[0]
f, axs = plt.subplots(1, n_subjects, figsize=(1.75*n_subjects, 2), dpi=dpi)
for (i, subject) in enumerate(np.unique(both_obj_df['subject'])):
    sns.lineplot(data=both_obj_df[both_obj_df['subject'] == subject], x='time', y='value', hue='goal',
                 ax=axs[i], hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']],
                 errorbar='se', err_kws={'lw': 0}, legend=None)
    axs[i].set(ylabel='Running speed (cm/s)', xlabel='Time from object entry (s)',
               title=subject, xticks=[-2, -1, 0, 1, 2, 3], yticks=[5, 10, 15, 20, 25])

plt.suptitle('Both objects')
sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(join(path_dict['google_drive_fig_path'], 'running_recordings_both_obj.pdf'))

# %%

f, axs = plt.subplots(1, n_subjects, figsize=(1.75*n_subjects, 2), dpi=dpi)
for (i, subject) in enumerate(np.unique(both_obj_df['subject'])):
    sns.lineplot(data=obj1_df[obj1_df['subject'] == subject], x='time', y='value', hue='goal',
                 ax=axs[i], hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']],
                 errorbar='se', err_kws={'lw': 0}, legend=None)
    axs[i].set(ylabel='Running speed (cm/s)', xlabel='Time from object entry (s)',
               title=subject, xticks=[-2, -1, 0, 1, 2, 3], yticks=[5, 10, 15, 20, 25])

plt.suptitle('First object')
sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(join(path_dict['google_drive_fig_path'], 'running_recordings_obj1.pdf'))

# %%

f, axs = plt.subplots(1, n_subjects, figsize=(1.75*n_subjects, 2), dpi=dpi)
for (i, subject) in enumerate(np.unique(both_obj_df['subject'])):
    sns.lineplot(data=obj2_df[obj2_df['subject'] == subject], x='time', y='value', hue='goal',
                 ax=axs[i], hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']],
                 errorbar='se', err_kws={'lw': 0}, legend=None)
    axs[i].set(ylabel='Running speed (cm/s)', xlabel='Time from object entry (s)',
               title=subject, xticks=[-2, -1, 0, 1, 2, 3], yticks=[5, 10, 15, 20, 25])

plt.suptitle('Second object')
sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(join(path_dict['google_drive_fig_path'], 'running_recordings_obj2.pdf'))
