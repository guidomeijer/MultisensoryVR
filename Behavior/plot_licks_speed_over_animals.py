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
from scipy.stats import ttest_1samp, wilcoxon
import pandas as pd
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style,
                            peri_multiple_events_time_histogram)

# Settings
T_BEFORE = [1, 0]
subject_dict = {'452505': '20231218',
                '459601': '20240328',
                '459602': '20240314',
                '452506': '20231211'}

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

# Loop over subjects
perc_incr_licking = np.empty(len(subject_dict))
perc_slowing = np.empty(len(subject_dict))
number_incr_licking = np.empty(len(subject_dict))
for i, subject in enumerate(subject_dict.keys()):

    # List sessions
    ses = subject_dict[subject]

    # Load in data
    trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'))
    lick_times = np.load(join(data_path, 'Subjects', subject, ses, 'lick.times.npy'))
    wheel_times = np.load(join(data_path, 'Subjects', subject, ses, 'continuous.times.npy'))
    wheel_speed = np.load(join(data_path, 'Subjects', subject,
                          ses, 'continuous.wheelSpeed.npy'))

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

    # Loop over trials and get speed and licks
    goal_licks = np.empty(goal_obj_enters.shape)
    goal_speed = np.empty(goal_obj_enters.shape)
    for j, this_enter in enumerate(goal_obj_enters):
        goal_licks[j] = np.sum((lick_times > this_enter - T_BEFORE[0])
                               & (lick_times < this_enter - T_BEFORE[1]))
        goal_speed[j] = np.mean(wheel_speed[(wheel_times > this_enter - T_BEFORE[0])
                                            & (wheel_times < this_enter - T_BEFORE[1])])
            
    nogoal_licks = np.empty(goal_obj_enters.shape)
    nogoal_speed = np.empty(goal_obj_enters.shape)
    for j, this_enter in enumerate(nogoal_obj_enters):
        nogoal_licks[j] = np.sum((lick_times > this_enter - T_BEFORE[0])
                                 & (lick_times < this_enter - T_BEFORE[1]))
        nogoal_speed[j] = np.nanmedian(wheel_speed[(wheel_times > this_enter - T_BEFORE[0])
                                                   & (wheel_times < this_enter - T_BEFORE[1])])
        
    # Calculate percentage slowing and anticipatory licking
    perc_incr_licking[i] = ((np.mean(goal_licks) - np.mean(nogoal_licks))
                            / np.mean(nogoal_licks)) * 100
    number_incr_licking[i] = np.mean(goal_licks) - np.mean(nogoal_licks)
    perc_slowing[i] = ((np.nanmedian(nogoal_speed) - np.nanmedian(goal_speed))
                       / np.nanmedian(nogoal_speed)) * 100 
    
# %% Stats
_, p_licks = ttest_1samp(number_incr_licking, 0)
_, p_running = ttest_1samp(perc_slowing, 0)

    
# %% Plot

df_results = pd.DataFrame(data={
    'number_incr_licking': number_incr_licking,
    'perc_slowing': perc_slowing})
df_results = df_results.melt()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 1.75), dpi=dpi)

sns.swarmplot(data=df_results[df_results['variable'] == 'perc_slowing'],
              x='variable', y='value', ax=ax1)
ax1.set(ylabel='Percentage decrease (%)', xlabel='', title='slowing',
        xticks=[0.5], yticks=[0, 10, 20, 30])
ax1.axes.get_xaxis().set_visible(False)
ax1.text(0, 25, '**', ha='center', va='center', fontsize=12)

sns.swarmplot(data=df_results[df_results['variable'] == 'number_incr_licking'],
              x='variable', y='value', ax=ax2)
ax2.set(ylabel='Extra licks per trial', xlabel='', title='licking',
        xticks=[0.5], yticks=[0, 0.5, 1])
ax2.axes.get_xaxis().set_visible(False)
ax2.text(0, 0.82, '**', ha='center', va='center', fontsize=12)

f.suptitle('Anticipatory', y=0.87, x=0.55)
sns.despine(trim=True)   
plt.tight_layout(pad=1.5)
plt.savefig(join(path_dict['fig_path'], 'licking_slowing_over_animals.pdf'))

    