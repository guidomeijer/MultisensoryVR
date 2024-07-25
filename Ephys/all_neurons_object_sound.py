# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:24:36 2024

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
BIN_SIZE = 0.025
SMOOTHING = 0.1
MIN_FR = 0.1
colors = {'goal': 'green', 'no-goal': 'red', 'control': 'gray'}

# Load in data
path_dict = paths(sync=False)

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE, histology=True, only_good=False)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))

# Get timestamps of entry of goal, no-goal, and control object sets for sound 1
goal_obj_enters_sound1 = trials.loc[trials['soundId'] == 1, 'enterObj1'].dropna().values
nogoal_obj_enters_sound1 = trials.loc[trials['soundId'] == 1, 'enterObj2'].dropna().values
control_obj_enters_sound1 = trials.loc[trials['soundId'] == 1, 'enterObj3'].dropna().values
all_obj_enters_sound1 = np.concatenate((goal_obj_enters_sound1, nogoal_obj_enters_sound1, control_obj_enters_sound1))
all_obj_ids_sound1 = np.concatenate((np.ones(goal_obj_enters_sound1.shape[0]), 
                                     np.ones(nogoal_obj_enters_sound1.shape[0]) * 2, 
                                     np.ones(control_obj_enters_sound1.shape[0]) * 3))


# Get timestamps of entry of goal, no-goal, and control object sets for sound 2
goal_obj_enters_sound2 = trials.loc[trials['soundId'] == 2, 'enterObj2'].dropna().values
nogoal_obj_enters_sound2 = trials.loc[trials['soundId'] == 2, 'enterObj1'].dropna().values
control_obj_enters_sound2 = trials.loc[trials['soundId'] == 2, 'enterObj3'].dropna().values
all_obj_enters_sound2 = np.concatenate((goal_obj_enters_sound2, nogoal_obj_enters_sound2, control_obj_enters_sound2))
all_obj_ids_sound2 = np.concatenate((np.ones(goal_obj_enters_sound2.shape[0]), 
                                     np.ones(nogoal_obj_enters_sound2.shape[0]) * 2,
                                     np.ones(control_obj_enters_sound2.shape[0]) * 3))

# %%
for i, neuron_id in enumerate(clusters['cluster_id']):
    if np.sum(spikes['clusters'] == neuron_id) / spikes['times'][-1] < 0.1:
        continue
    
    # Plot neurons for sound 1 and sound 2
    _, dpi = figure_style(font_size=9)
    
    # First get y limits of both plots
    f, (ax1, ax2) = plt.subplots(1, 2)
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj_enters_sound1, all_obj_ids_sound1, [neuron_id],
        t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax1,
        pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}, {'color': colors['control'], 'lw': 1}],
        errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
        raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}, {'color': colors['control'], 'lw': 0.5}],
        smoothing = SMOOTHING, include_raster=True
        )
    _, y_max_1 = ax1.get_ylim()
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj_enters_sound2, all_obj_ids_sound2, [neuron_id],
        t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=ax2,
        pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}, {'color': colors['control'], 'lw': 1}],
        errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
        raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}, {'color': colors['control'], 'lw': 0.5}],
        smoothing=SMOOTHING, include_raster=True
        )
    _, y_max_2 = ax2.get_ylim()
    plt.close(f)
    y_max_use = np.ceil(np.max([y_max_1, y_max_2]))
    
    # Now plot figure
    fig, axes = plt.subplots(1, 2, figsize=(3, 2), dpi=dpi, sharex=True, sharey=False)
    
    # Plot for sound 1
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj_enters_sound1, all_obj_ids_sound1,
        [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=axes[0],
        smoothing = SMOOTHING, ylim=y_max_use,
        pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}, {'color': colors['control'], 'lw': 1}],
        errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
        raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}, {'color': colors['control'], 'lw': 0.5}],
        eventline_kwargs={'lw': 0}, include_raster=True
    )
    axes[0].set_title('Sound 1')
    axes[0].set(ylabel='Firing rate (spks/s)')
    axes[0].yaxis.set_label_coords(-0.2, 0.75)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[0].plot([0, 0], axes[0].get_ylim(), color='r', linestyle='--', lw=0.5, clip_on=False, alpha=0.5)
    axes[0].set(yticks=[0, y_max_use], xticks=[-1, 0, 0.5, 1, 2], xlabel=' ')
    
    # Plot for sound 2
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj_enters_sound2, all_obj_ids_sound2,
        [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, ax=axes[1],
        smoothing=SMOOTHING, ylim=y_max_use,
        pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}, {'color': colors['control'], 'lw': 1}],
        errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
        raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}, {'color': colors['control'], 'lw': 0.5}],
        eventline_kwargs={'lw': 0}, include_raster=True
    )
    axes[1].set_title('Sound 2')
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[1].plot([0, 0], [axes[1].get_ylim()[0], y_max_use], color='r', linestyle='--', lw=0.5, clip_on=False, alpha=0.5)
    axes[1].set(xlabel='', ylabel='')
    axes[1].set(yticks=[0, np.round(y_max_use)], xticks=[-1, 0, 1, 2])
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    fig.text(0.5, 0.06, 'Time from object entry (s)', ha='center')
    
    # Adjust layout and save figure
    plt.suptitle(clusters['acronym'][clusters['cluster_id'] == neuron_id][0])
    plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}', 'ObjectSound',
                     f'{SUBJECT}_{DATE}_{PROBE}_neuron{neuron_id}.jpg'), dpi=150)
    plt.close(fig)