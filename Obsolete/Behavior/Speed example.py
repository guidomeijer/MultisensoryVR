# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:33:01 2024

@author: zayel
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from os.path import join, isfile
import pandas as pd
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style,
                            peri_multiple_events_time_histogram)

# Settings
T_BEFORE = 2
T_AFTER = 3
BIN_SIZE = 0.2
SMOOTHING = 0.1
SUBJECT = '459601'  # Manually specify the subject
SESSION = '20240328'  # Manually specify the session

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

# Load session data
session_path = join(data_path, 'Subjects', SUBJECT, SESSION)

# Load data
trials = pd.read_csv(join(session_path, 'trials.csv'))
wheel_times = np.load(join(session_path, 'continuous.times.npy'))
wheel_speed = np.load(join(session_path, 'continuous.wheelSpeed.npy'))

# Get timestamps of entry of goal, no-goal and control object sets
goal_obj_enters = np.concatenate((
    trials.loc[trials['soundId'] == 1, 'enterObj1'],
    trials.loc[trials['soundId'] == 2, 'enterObj2']))
nogoal_obj_enters = np.concatenate((
    trials.loc[trials['soundId'] == 1, 'enterObj2'],
    trials.loc[trials['soundId'] == 2, 'enterObj1']))
control_obj_enters = trials['enterObj3'].values
all_obj_enters = np.concatenate((goal_obj_enters, nogoal_obj_enters, control_obj_enters))
all_obj_ids = np.concatenate(
    (np.ones(goal_obj_enters.shape[0]),
     np.ones(nogoal_obj_enters.shape[0]) * 2,
     np.ones(control_obj_enters.shape[0]) * 3))
all_obj_ids = all_obj_ids[~np.isnan(all_obj_enters)]
all_obj_enters = all_obj_enters[~np.isnan(all_obj_enters)]

# Downsample wheel speed and convert to cm/s
wheel_speed = wheel_speed[::50] / 10
wheel_times = wheel_times[::50]

# Create figure
f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

# Plot running speed
peri_event_trace(wheel_speed, wheel_times, all_obj_enters,
                 event_ids=all_obj_ids,
                 color_palette=[colors['goal'], colors['no-goal'], colors['control']],
                 event_labels=['Goal', 'Distractor', 'Control'],
                 t_before=T_BEFORE, t_after=T_AFTER, ax=ax, kwargs={'zorder': 1})

ax.set(ylabel='Speed (cm/s)', xticks=np.arange(-T_BEFORE, T_AFTER + 1),
       title='Running speed', xlabel='Time from object entry (s)')

# Customize font sizes
ax.set_xlabel('Time from object entry (s)')
ax.set_ylabel('Speed (cm/s)')

# Add a vertical line at time = 0
max_y = ax.get_ylim()[1]
ax.plot([0, 0], [0, np.ceil(np.max(max_y))], color='gray', ls='--', lw=0.75, zorder=0)
    
# Remove the legend
ax.legend().set_visible(False)

# Finalize figure
sns.despine(trim=True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(join(path_dict['fig_path'], 'Speed example.jpg'), dpi=600)
plt.savefig(join(path_dict['fig_path'], 'Speed example.pdf'))
plt.show()
