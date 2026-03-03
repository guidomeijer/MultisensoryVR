# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:24:36 2024

@author: Guido & Zayel
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, isdir
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from msvr_functions import (paths, peri_multiple_events_time_histogram,
                            load_neural_data, figure_style, load_subjects)

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
HISTOLOGY = True
MIN_SPEED = 50  # mm/s
MIN_FR = 1

# Time
T_BEFORE = 2  # s
T_AFTER = 2
T_BIN_SIZE = 0.025
T_SMOOTHING = 0.1

# Distance
D_BEFORE = 10  # cm
D_AFTER = 160
D_BIN_SIZE = 1
D_SMOOTHING = 2

colors, dpi = figure_style()

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE, histology=True, only_good=True)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))
wheel_speed = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.wheelSpeed.npy'))
wheel_dist = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.wheelDistance.npy'))
wheel_times = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.times.npy'))

# For the distance plots, set a speed threshold
# Find for each spike its corresponding speed
indices = np.searchsorted(wheel_times, spikes['times'], side='right') - 1
indices = np.clip(indices, 0, wheel_dist.shape[0] - 1)
spike_speed = wheel_speed[indices]
spikes_dist = spikes['distances'][spike_speed >= MIN_SPEED]
clusters_dist = spikes['clusters'][spike_speed >= MIN_SPEED]

# %% Prepare data

# Get reward contingencies
sound1_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'Sound1Obj'].values[0]
sound2_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'Sound2Obj'].values[0]
control_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'ControlObject'].values[0]

obj1_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 1)[0][0] + 1
obj2_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 2)[0][0] + 1
obj3_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 3)[0][0] + 1

# Get timestamps of entry of goal, no-goal, and control object sets for sound 1
goal_obj_enters_sound1 = trials.loc[trials['soundId'] == 1, f'enterObj{sound1_obj}'].dropna().values
nogoal_obj_enters_sound1 = trials.loc[trials['soundId'] == 1, f'enterObj{sound2_obj}'].dropna().values
control_obj_enters_sound1 = trials.loc[trials['soundId'] == 1, f'enterObj{control_obj}'].dropna().values
all_obj_enters_sound1 = np.concatenate((goal_obj_enters_sound1, nogoal_obj_enters_sound1, control_obj_enters_sound1))
all_obj_ids_sound1 = np.concatenate((np.ones(goal_obj_enters_sound1.shape[0]), 
                                     np.ones(nogoal_obj_enters_sound1.shape[0]) * 2, 
                                     np.ones(control_obj_enters_sound1.shape[0]) * 3))


# Get timestamps of entry of goal, no-goal, and control object sets for sound 2
goal_obj_enters_sound2 = trials.loc[trials['soundId'] == 2, f'enterObj{sound1_obj}'].dropna().values
nogoal_obj_enters_sound2 = trials.loc[trials['soundId'] == 2, f'enterObj{sound2_obj}'].dropna().values
control_obj_enters_sound2 = trials.loc[trials['soundId'] == 2, f'enterObj{control_obj}'].dropna().values
all_obj_enters_sound2 = np.concatenate((goal_obj_enters_sound2, nogoal_obj_enters_sound2, control_obj_enters_sound2))
all_obj_ids_sound2 = np.concatenate((np.ones(goal_obj_enters_sound2.shape[0]), 
                                     np.ones(nogoal_obj_enters_sound2.shape[0]) * 2,
                                     np.ones(control_obj_enters_sound2.shape[0]) * 3))

# Object 1 entries conditioned on whether the object is a target or distractor
obj1_sound1 = trials.loc[trials['soundId'] == 1, 'enterObj1'].dropna().values
obj1_sound2 = trials.loc[trials['soundId'] == 2, 'enterObj1'].dropna().values
all_obj1_enters = np.concatenate((obj1_sound1, obj1_sound2))
all_obj1_ids = np.concatenate((np.ones(obj1_sound1.shape[0]), 
                               np.ones(obj1_sound2.shape[0]) * 2))
obj2_sound1 = trials.loc[trials['soundId'] == 1, 'enterObj2'].dropna().values
obj2_sound2 = trials.loc[trials['soundId'] == 2, 'enterObj2'].dropna().values
all_obj2_enters = np.concatenate((obj2_sound1, obj2_sound2))
all_obj2_ids = np.concatenate((np.ones(obj2_sound1.shape[0]), 
                               np.ones(obj2_sound2.shape[0]) * 2))
obj3_sound1 = trials.loc[trials['soundId'] == 1, 'enterObj3'].dropna().values
obj3_sound2 = trials.loc[trials['soundId'] == 2, 'enterObj3'].dropna().values
all_obj3_enters = np.concatenate((obj3_sound1, obj3_sound2))
all_obj3_ids = np.concatenate((np.ones(obj3_sound1.shape[0]), 
                               np.ones(obj3_sound2.shape[0]) * 2))

# %% PLot neurons

# Create folders if necessary
if not isdir(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}')):
    os.mkdir(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}'))
if not isdir(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}', 'SoundTarget')):
    os.mkdir(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}', 'SoundTarget'))
if not isdir(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}', 'ObjectSound')):
    os.mkdir(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}', 'ObjectSound'))
if not isdir(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}', 'EnvironmentSound')):
    os.mkdir(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}', 'EnvironmentSound'))

for i, neuron_id in enumerate(clusters['cluster_id']):
    if np.sum(spikes['clusters'] == neuron_id) / spikes['times'][-1] < MIN_FR:
        continue
    print(f'Plotting neuron {i} of {clusters["cluster_id"].shape[0]}')
    
    if HISTOLOGY:
        # Get region
        region = clusters['acronym'][clusters['cluster_id'] == neuron_id][0]
        region = region.replace('/', '-')
    else:
        region = 'root'
    
    # %% Plot object entries for sound 1 and sound 2
    
    # First get y limits of both plots
    f, (ax1, ax2) = plt.subplots(1, 2)
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj_enters_sound1, all_obj_ids_sound1, [neuron_id],
        t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax1,
        pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}, {'color': colors['control'], 'lw': 1}],
        errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
        raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}, {'color': colors['control'], 'lw': 0.5}],
        smoothing = T_SMOOTHING, include_raster=True
        )
    _, y_max_1 = ax1.get_ylim()
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj_enters_sound2, all_obj_ids_sound2, [neuron_id],
        t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax2,
        pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}, {'color': colors['control'], 'lw': 1}],
        errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
        raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}, {'color': colors['control'], 'lw': 0.5}],
        smoothing=T_SMOOTHING, include_raster=True
        )
    _, y_max_2 = ax2.get_ylim()
    plt.close(f)
    y_max_use = np.ceil(np.max([y_max_1, y_max_2]))
    
    # Now plot figure
    fig, axes = plt.subplots(1, 2, figsize=(2.5, 2), dpi=dpi, sharex=True, sharey=False)
    
    # Plot for sound 1
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj_enters_sound1, all_obj_ids_sound1,
        [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=axes[0],
        smoothing = T_SMOOTHING, ylim=y_max_use,
        pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}, {'color': colors['control'], 'lw': 1}],
        errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
        raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}, {'color': colors['control'], 'lw': 0.5}],
        eventline_kwargs={'lw': 0}, include_raster=True
    )
    axes[0].set_title('Sound 1')
    axes[0].set(ylabel='Firing rate (spks/s)')
    axes[0].yaxis.set_label_coords(-0.3, 0.75)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[0].plot([0, 0], axes[0].get_ylim(), color='r', linestyle='--', lw=0.5, clip_on=False, alpha=0.5)
    axes[0].set(yticks=[0, y_max_use], xticks=[-1, 0, 0.5, 1, 2], xlabel=' ')
    
    # Plot for sound 2
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj_enters_sound2, all_obj_ids_sound2,
        [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=axes[1],
        smoothing=T_SMOOTHING, ylim=y_max_use,
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
    plt.suptitle(region)
    plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}', 'SoundTarget',
                     f'{region}_{DATE}_{PROBE}_neuron{neuron_id}.jpg'), dpi=300)
    plt.close(fig)
    
    # %% Plot object conditioned on sound
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4, 2), dpi=dpi)
    
    if obj1_goal_sound == 1:
        color_1 = colors['goal']
        color_2 = colors['no-goal']
    elif obj1_goal_sound == 2:
        color_1 = colors['no-goal']
        color_2 = colors['goal']
    elif obj1_goal_sound == 3:
        color_1 = colors['sound1']
        color_2 = colors['sound2']    
        
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj1_enters, all_obj1_ids,
        [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax1,
        smoothing = T_SMOOTHING, ylim=y_max_use,
        pethline_kwargs=[{'color': color_1, 'lw': 1}, {'color': color_2, 'lw': 1}],
        errbar_kwargs=[{'color': color_1, 'alpha': 0.3, 'lw': 0}, {'color': color_2, 'alpha': 0.3, 'lw': 0}],
        raster_kwargs=[{'color': color_1, 'lw': 0.5}, {'color': color_2, 'lw': 0.5}],
        eventline_kwargs={'lw': 0}, include_raster=True)
    ax1.set(title='Object 1', ylabel='Firing rate (spks/s)', yticks=[0, np.ceil(ax1.get_ylim()[1])],
            xlabel='')
    ax1.yaxis.set_label_coords(-0.3, 0.75)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    if obj2_goal_sound == 1:
        color_1 = colors['goal']
        color_2 = colors['no-goal']
    elif obj2_goal_sound == 2:
        color_1 = colors['no-goal']
        color_2 = colors['goal']
    elif obj2_goal_sound == 3:
        color_1 = colors['sound1']
        color_2 = colors['sound2']    
        
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj2_enters, all_obj2_ids,
        [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax2,
        smoothing = T_SMOOTHING, ylim=y_max_use,
        pethline_kwargs=[{'color': color_1, 'lw': 1}, {'color': color_2, 'lw': 1}],
        errbar_kwargs=[{'color': color_1, 'alpha': 0.3, 'lw': 0}, {'color': color_2, 'alpha': 0.3, 'lw': 0}],
        raster_kwargs=[{'color': color_1, 'lw': 0.5}, {'color': color_2, 'lw': 0.5}],
        eventline_kwargs={'lw': 0}, include_raster=True)
    ax2.set(title='Object 2', yticks=[0, np.ceil(ax2.get_ylim()[1])], ylabel='', xlabel='')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    if obj3_goal_sound == 1:
        color_1 = colors['goal']
        color_2 = colors['no-goal']
    elif obj3_goal_sound == 2:
        color_1 = colors['no-goal']
        color_2 = colors['goal']
    elif obj3_goal_sound == 3:
        color_1 = colors['sound1']
        color_2 = colors['sound2']    
        
    peri_multiple_events_time_histogram(
        spikes['times'], spikes['clusters'], all_obj3_enters, all_obj3_ids,
        [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax3,
        smoothing = T_SMOOTHING, ylim=y_max_use,
        pethline_kwargs=[{'color': color_1, 'lw': 1}, {'color': color_2, 'lw': 1}],
        errbar_kwargs=[{'color': color_1, 'alpha': 0.3, 'lw': 0}, {'color': color_2, 'alpha': 0.3, 'lw': 0}],
        raster_kwargs=[{'color': color_1, 'lw': 0.5}, {'color': color_2, 'lw': 0.5}],
        eventline_kwargs={'lw': 0}, include_raster=True)
    ax3.set(title='Object 3', yticks=[0, np.ceil(ax3.get_ylim()[1])], ylabel='', xlabel='')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    f.text(0.5, 0.05, 'Time from object entry (s)', ha='center')
    plt.suptitle(region)
    plt.subplots_adjust(bottom=0.2, top=0.8)
    plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}', 'ObjectSound',
                     f'{region}_{DATE}_{PROBE}_neuron{neuron_id}.jpg'), dpi=300)
    plt.close(f)
    
    # %% Plot activity over entire environment split by sound
    sound_colors = [colors['sound1'], colors['sound2'], colors['control']]
    
    f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
    
    peri_multiple_events_time_histogram(spikes_dist / 10, clusters_dist,
                              trials['enterEnvPos'] / 10, trials['soundId'], [neuron_id],
                              t_before=D_BEFORE, t_after=D_AFTER, bin_size=D_BIN_SIZE, 
                              smoothing=D_SMOOTHING, ax=ax1,
                              pethline_kwargs=[{'color': colors['sound1'], 'lw': 1},
                                               {'color': colors['sound2'], 'lw': 1}],
                              errbar_kwargs=[{'color': colors['sound1'], 'alpha': 0.3, 'lw': 0},
                                             {'color': colors['sound2'], 'alpha': 0.3, 'lw': 0}],
                              raster_kwargs=[{'color': colors['sound1'], 'lw': 0.5},
                                             {'color': colors['sound2'], 'lw': 0.5}],
                              eventline_kwargs={'lw': 0}, include_raster=True)
    y_max = ax1.get_ylim()[1]
    if y_max > 1.95:
        y_max = int(np.ceil(y_max))
    else:
        y_max = np.round(y_max + 0.1, decimals=1) 
        
    ax1.set(ylabel='Firing rate (spks/cm)', yticks=[0, y_max],
            yticklabels=[0, y_max],
            xlabel='Distance from environment entry (cm)', title=region)
    ax1.yaxis.set_label_coords(-0.175, 0.75)
    """
    ax1.plot([47.5, 47.5], ax1.get_ylim(), color=sound_colors[obj1_goal_sound-1],
             linestyle='--', lw=0.5, clip_on=False, alpha=0.5)
    ax1.plot([47.5, 47.5], ax1.get_ylim(), color=sound_colors[obj1_goal_sound-1],
             linestyle='--', lw=0.5, clip_on=False, alpha=0.5)
    ax1.plot([47.5, 47.5], ax1.get_ylim(), color=sound_colors[obj1_goal_sound-1],
             linestyle='--', lw=0.5, clip_on=False, alpha=0.5)
    """
    
    plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', f'{SUBJECT}', 'EnvironmentSound',
                     f'{region}_{DATE}_{PROBE}_neuron{neuron_id}.jpg'), dpi=300)
    
    #plt.close(f)
    
    
    
    