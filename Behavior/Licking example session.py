# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:46:55 2024

@author: zayel
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
T_BEFORE = 2
T_AFTER = 3
T_PLOT = [-2, 3]
BIN_SIZE = 0.2
SMOOTHING = 0.1
WIN_SIZE = 2  # s
WIN_SHIFT = 0.05  # s
FS = 1000  # sampling rate
FREQ = [5, 10]
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
sessions = join(data_path, 'Subjects', SUBJECT, SESSION)
# Load data
trials = pd.read_csv(join(sessions, 'trials.csv'))
lick_times = np.load(join(sessions, 'lick.times.npy'))

# Set font sizes
plt.rcParams.update({'font.size': 12})  # Global font size
title_fontsize = 7
label_fontsize = 7
tick_fontsize = 7

# Create lick figure
f, ax = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)

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

peri_multiple_events_time_histogram(
    lick_times, np.ones(lick_times.shape[0]), all_obj_enters, all_obj_ids,
    [1], t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, smoothing=SMOOTHING, ax=ax,
    pethline_kwargs=[{'color': colors['goal'], 'lw': 1},
                     {'color': colors['no-goal'], 'lw': 1},
                     {'color': colors['control'], 'lw': 1}],
    errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0},
                   {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0},
                   {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': colors['goal'], 'lw': 0.65},
                   {'color': colors['no-goal'], 'lw': 0.65},
                   {'color': colors['control'], 'lw': 0.65}],
    eventline_kwargs={'lw': 0}, include_raster=True)

# Set labels and title
ax.set(xticks=np.arange(-T_BEFORE, T_AFTER+1),
       yticks=[0, np.ceil(ax.get_ylim()[1])])
ax.yaxis.set_label_coords(-0.1, 0.75)
max_y = ax.get_ylim()[1]

# Place the dotted line now we know the y lim extend
ax.plot([0, 0], [-max_y, np.ceil(np.max(max_y))], color='gray', ls='--', lw=0.75, zorder=0)
sns.despine(trim=True)

# Customize font sizes
ax.set_title('Licks', fontsize=title_fontsize)
ax.set_xlabel('Time from object entry (s)', fontsize=label_fontsize)
ax.set_ylabel('Licks/s', fontsize=label_fontsize)
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax.yaxis.set_label_coords(-0.1, 0.75)

# Adjust layout to ensure labels are not clipped
plt.tight_layout()

# Save figure
plt.savefig(join(path_dict['fig_path'], 'Lick example2.jpg'), dpi=600)
plt.savefig(join(path_dict['fig_path'], 'Lick example.pdf'), dpi=600)

# Create a separate figure for the legend
legend_labels = ['Goal', 'No-goal', 'Control']
legend_colors = [colors['goal'], colors['no-goal'], colors['control']]
legend_lines = [plt.Line2D([0], [0], color=legend_colors[i], lw=1) for i in range(3)]

# Create a new figure for the legend
legend_fig, legend_ax = plt.subplots(figsize=(2, 2), dpi=dpi)
legend_ax.legend(legend_lines, legend_labels, loc='center', frameon=False)
legend_ax.axis('off')  # Hide the axis
plt.savefig(join(path_dict['fig_path'], 'Lick speed legend.jpg'), dpi=600)
plt.savefig(join(path_dict['fig_path'], 'Lick speed legend.pdf'), dpi=600)
