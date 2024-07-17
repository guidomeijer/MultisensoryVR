# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:39:04 2024

@author: zayel
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from msvr_functions import (paths, peri_multiple_events_time_histogram,
                            load_neural_data, figure_style)


# Settings
SUBJECT = '459602'
DATE = '20240315'
PROBE = 'probe00'
T_BEFORE = 1
T_AFTER = 2
#NEURON_ID = 246  # sound and reward
#NEURON_ID = 363  # sound
NEURON_ID = 392  # reward
TITLE = 'Reward neuron'
BIN_SIZE = 0.025
colors = {'goal': 'green', 'no-goal': 'red', 'control': 'gray'}
dpi = 300

# Load in data
path_dict = paths(sync=False)

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE, histology=False, only_good=True)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))
spike_times = spikes['times'][spikes['clusters'] == NEURON_ID]

# Get timestamps of entry of goal, no-goal, and control object sets
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
     np.ones(nogoal_obj_enters.shape[0])*2,
     np.ones(control_obj_enters.shape[0])*3))
all_obj_ids = all_obj_ids[~np.isnan(all_obj_enters)]
all_obj_enters = all_obj_enters[~np.isnan(all_obj_enters)]
sort_idx = np.argsort(all_obj_enters)
all_obj_times = all_obj_enters[sort_idx]
all_obj_times_ids = all_obj_ids[sort_idx]

# Plot neurons
_, dpi = figure_style(font_size=9)
f, ax = plt.subplots(figsize=(2, 1.75), dpi = dpi)
peri_multiple_events_time_histogram(
    spike_times, np.ones(spike_times.shape[0]), all_obj_enters, all_obj_ids,
            [1], t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax,
                          pethline_kwargs=[{'color': colors['goal'], 'lw': 1},
                                           {'color': colors['no-goal'], 'lw': 1},
                                           {'color': colors['control'], 'lw': 1}],
                          errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0},
                                         {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0},
                                         {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
                          raster_kwargs=[{'color': colors['goal'], 'lw': 0.5},
                                         {'color': colors['no-goal'], 'lw': 0.5},
                                         {'color': colors['control'], 'lw': 0.5}],
                          eventline_kwargs={'lw': 0}, include_raster=True)

# Calculate the y-axis range
y_min, y_max = ax.get_ylim()
y_range = y_max - y_min

# Adjust the length of the lines based on the y-axis range
line_length = 0.5 * y_range  # You can adjust the multiplier as needed
ax.plot([0, 0], [-line_length, line_length], color='gray', linestyle='--', lw=0.5, clip_on=False, alpha=0.5)
ax.set(
       yticks=[np.round(ax.get_ylim()[1])], xticks=[-1, 0, 1, 2],
       ylim=[ax.get_ylim()[0], np.round(ax.get_ylim()[1])])
ax.set_title(f'{TITLE}', loc='center')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# Remove axis labels
ax.set_xlabel('')
ax.set_ylabel('')

plt.tight_layout()
plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', f'{TITLE}.pdf'), dpi=600)