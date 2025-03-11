# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:24:36 2024

@author: Guido & Zayel
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, isdir, isfile
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from msvr_functions import (paths, peri_multiple_events_time_histogram, load_objects,
                            load_neural_data, figure_style, load_subjects)

# Settings
OVERWITE = False
MIN_SPEED = 50  # mm/s

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
rec_df = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
neuron_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
neuron_df['subject'] = neuron_df['subject'].astype(str)
neuron_df['date'] = neuron_df['date'].astype(str)
neuron_df = neuron_df[neuron_df['region'] != 'root']

# Create folders if necessary
if not isdir(join(path_dict['fig_path'], 'ExampleNeurons', 'GoalNeurons')):
    os.mkdir(join(path_dict['fig_path'], 'ExampleNeurons', 'GoalNeurons'))

# Loop over recordings
for i, (subject, date, probe) in enumerate(zip(rec_df['subject'], rec_df['date'], rec_df['probe'])):
    print(f'{subject}', f'{date}', f'{probe}')
    
    # Load in data
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe, histology=True, only_good=True)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)
    wheel_speed = np.load(join(path_dict['local_data_path'], 'Subjects', subject, date, 'continuous.wheelSpeed.npy'))
    wheel_dist = np.load(join(path_dict['local_data_path'], 'Subjects', subject, date, 'continuous.wheelDistance.npy'))
    wheel_times = np.load(join(path_dict['local_data_path'], 'Subjects', subject, date, 'continuous.times.npy'))
        
    # Set a speed threshold, find for each spike its corresponding speed
    indices = np.searchsorted(wheel_times, spikes['times'], side='right') - 1
    indices = np.clip(indices, 0, wheel_dist.shape[0] - 1)
    spike_speed = wheel_speed[indices]
    spikes_dist = spikes['distances'][spike_speed >= MIN_SPEED]
    clusters_dist = spikes['clusters'][spike_speed >= MIN_SPEED]

    # Convert from mm to cm
    spikes_dist = spikes_dist / 10
    trials['enterEnvPos'] = trials['enterEnvPos'] / 10

    
    # %% Goal neurons
    
    these_neurons = neuron_df[(neuron_df['subject'] == subject)
                              & (neuron_df['date'] == date)
                              & (neuron_df['probe'] == probe)
                              & (neuron_df['sig_goal'])]
    
    for i, neuron_id in enumerate(these_neurons['neuron_id']):
      
        # Get region
        region = clusters['region'][clusters['cluster_id'] == neuron_id][0]
       
        # Plot object conditioned on sound
        if not isdir(join(path_dict['fig_path'], 'ExampleNeurons', 'GoalNeurons', f'{region}')):
            os.mkdir(join(path_dict['fig_path'], 'ExampleNeurons', 'GoalNeurons', f'{region}'))
        if isfile(join(path_dict['fig_path'], 'ExampleNeurons', 'GoalNeurons', f'{region}',
                       f'{subject}_{date}_{probe}_neuron{neuron_id}.jpg')):
            continue
        print(f'Plotting neuron {i} of {these_neurons.shape[0]}')
        
        # First get y limits of both plots
        try:
            f, (ax1, ax2) = plt.subplots(1, 2)
            peri_multiple_events_time_histogram(
                spikes['times'], spikes['clusters'], 
                all_obj_df.loc[all_obj_df['object'] == 1, 'times'], 
                all_obj_df.loc[all_obj_df['object'] == 1, 'goal'],
                [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax1,
                pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}, {'color': colors['control'], 'lw': 1}],
                errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
                raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}, {'color': colors['control'], 'lw': 0.5}],
                smoothing = T_SMOOTHING, include_raster=True
                )
            _, y_max_1 = ax1.get_ylim()
            peri_multiple_events_time_histogram(
                spikes['times'], spikes['clusters'], 
                all_obj_df.loc[all_obj_df['object'] == 2, 'times'], 
                all_obj_df.loc[all_obj_df['object'] == 2, 'goal'],
                [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax2,
                pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}, {'color': colors['control'], 'lw': 1}],
                errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
                raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}, {'color': colors['control'], 'lw': 0.5}],
                smoothing=T_SMOOTHING, include_raster=True
                )
            _, y_max_2 = ax2.get_ylim()
            plt.close(f)
            y_max_use = np.ceil(np.max([y_max_1, y_max_2]))
        except Exception:
            continue
        
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4, 2), dpi=dpi)
 
        peri_multiple_events_time_histogram(
            spikes['times'], spikes['clusters'],
            all_obj_df.loc[all_obj_df['object'] == 1, 'times'],
            all_obj_df.loc[all_obj_df['object'] == 1, 'goal'],
            [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax1,
            smoothing = T_SMOOTHING, ylim=y_max_use,
            pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}],
            errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}],
            raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}],
            eventline_kwargs={'lw': 0}, include_raster=True)
        ax1.set(title='Object 1', ylabel='Firing rate (spks/s)', yticks=[0, np.ceil(ax1.get_ylim()[1])],
                xlabel='')
        ax1.yaxis.set_label_coords(-0.3, 0.75)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            
        peri_multiple_events_time_histogram(
            spikes['times'], spikes['clusters'], 
            all_obj_df.loc[all_obj_df['object'] == 2, 'times'],
            all_obj_df.loc[all_obj_df['object'] == 2, 'goal'],
            [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax2,
            smoothing = T_SMOOTHING, ylim=y_max_use,
            pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}],
            errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}],
            raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}],
            eventline_kwargs={'lw': 0}, include_raster=True)
        ax2.set(title='Object 2', yticks=[0, np.ceil(ax2.get_ylim()[1])], ylabel='', xlabel='')
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            
        peri_multiple_events_time_histogram(
            spikes['times'], spikes['clusters'], 
            all_obj_df.loc[all_obj_df['object'] == 3, 'times'],
            all_obj_df.loc[all_obj_df['object'] == 3, 'sound'],
            [neuron_id], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax3,
            smoothing = T_SMOOTHING, ylim=y_max_use,
            pethline_kwargs=[{'color': colors['sound1'], 'lw': 1}, {'color': colors['sound2'], 'lw': 1}],
            errbar_kwargs=[{'color': colors['sound1'], 'alpha': 0.3, 'lw': 0}, {'color': colors['sound2'], 'alpha': 0.3, 'lw': 0}],
            raster_kwargs=[{'color': colors['sound1'], 'lw': 0.5}, {'color': colors['sound2'], 'lw': 0.5}],
            eventline_kwargs={'lw': 0}, include_raster=True)
        ax3.set(title='Object 3', yticks=[0, np.ceil(ax3.get_ylim()[1])], ylabel='', xlabel='')
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        f.text(0.5, 0.05, 'Time from object entry (s)', ha='center')
        plt.suptitle(region)
        plt.subplots_adjust(bottom=0.2, top=0.8)
        
        # Save
        plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', 'GoalNeurons', f'{region}',
                         f'{subject}_{date}_{probe}_neuron{neuron_id}.jpg'), dpi=300)
        plt.close(f)
        
    # %% Environment
    
    these_neurons = neuron_df[(neuron_df['subject'] == subject)
                              & (neuron_df['date'] == date)
                              & (neuron_df['probe'] == probe)
                              & (neuron_df['sig_obj_onset'])]
    
    for i, neuron_id in enumerate(these_neurons['neuron_id']):
      
        # Get region
        region = clusters['region'][clusters['cluster_id'] == neuron_id][0]
       
        # Plot object conditioned on sound
        if not isdir(join(path_dict['fig_path'], 'ExampleNeurons', 'Environment', f'{region}')):
            os.mkdir(join(path_dict['fig_path'], 'ExampleNeurons', 'Environment', f'{region}'))
        if isfile(join(path_dict['fig_path'], 'ExampleNeurons', 'Environment', f'{region}',
                       f'{subject}_{date}_{probe}_neuron{neuron_id}.jpg')):
            continue
        print(f'Plotting neuron {i} of {these_neurons.shape[0]}')
        
        f, ax1 = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi)
        
        peri_multiple_events_time_histogram(spikes_dist, clusters_dist,
                                  trials['enterEnvPos'], trials['soundId'], [neuron_id],
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
        plt.tight_layout()
        
        # Save
        plt.savefig(join(path_dict['fig_path'], 'ExampleNeurons', 'Environment', f'{region}',
                         f'{subject}_{date}_{probe}_neuron{neuron_id}.jpg'), dpi=300)
        plt.close(f)