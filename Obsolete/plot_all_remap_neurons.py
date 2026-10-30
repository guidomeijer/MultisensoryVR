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
from msvr_functions import (paths, peri_multiple_events_time_histogram, load_objects, load_trials,
                            load_neural_data, figure_style, load_subjects)

# Settings
MIN_SPEED = 20  # mm/s
D_BEFORE_OBJ = 45
D_AFTER_OBJ = 0
D_BIN_SIZE = 0.5
D_SMOOTHING = 1

colors, dpi = figure_style()

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
rec_df = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# Load in data
neuron_df = pd.read_csv(path_dict['save_path'] / 'remapping_neurons.csv',
                        dtype={'subject': str, 'date': str})
neuron_df = neuron_df[neuron_df['region'] != 'root']

# Loop over recordings
for i, (subject, date, probe) in enumerate(zip(rec_df['subject'], rec_df['date'], rec_df['probe'])):
    if subject not in neuron_df['subject'].values:
        continue
    print(f'{subject}', f'{date}', f'{probe}')
    
    # Load in data
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe, histology=True, only_good=True)
    trials = load_trials(subject, date)
    all_obj_df = load_objects(subject, date)
    all_obj_df.loc[all_obj_df['goal'] == 0, 'goal'] = 2
    wheel_speed = np.load(join(path_dict['local_data_path'], 'Subjects', subject, date, 'continuous.wheelSpeed.npy'))
    wheel_dist = np.load(join(path_dict['local_data_path'], 'Subjects', subject, date, 'continuous.wheelDistance.npy'))
    wheel_times = np.load(join(path_dict['local_data_path'], 'Subjects', subject, date, 'continuous.times.npy'))
        
    # Set a speed threshold
    spikes_dist = spikes['distances'][spikes['speeds'] >= MIN_SPEED]
    clusters_dist = spikes['clusters'][spikes['speeds'] >= MIN_SPEED]

    # Convert from mm to cm
    spikes_dist = spikes_dist / 10
    all_obj_df['distances'] = all_obj_df['distances'] / 10
    
    # %% Goal neurons
    these_neurons = neuron_df[(neuron_df['subject'] == subject)
                              & (neuron_df['date'] == date)
                              & (neuron_df['probe'] == probe)
                              & (neuron_df['p_obj2'] < 0.05)]
    
    for i, neuron_id in enumerate(these_neurons['neuron_id']):
      
        # Get region
        region = clusters['region'][clusters['cluster_id'] == neuron_id][0]
        allen_acronym = clusters['acronym'][clusters['cluster_id'] == neuron_id][0]
       
        # Plot
        if not isdir(join(path_dict['fig_path'], 'RemapNeurons', f'{region}')):
            os.mkdir(join(path_dict['fig_path'], 'RemapNeurons', f'{region}'))
        
        if not isfile(join(path_dict['fig_path'], 'RemapNeurons', f'{region}',
                       f'{subject}_{date}_{probe}_neuron{neuron_id}.jpg')):
            
            fig, ax1 = plt.subplots(figsize=(2, 2), dpi=dpi)
     
            peri_multiple_events_time_histogram(
                spikes_dist, clusters_dist, 
                all_obj_df.loc[all_obj_df['object'] == 2, 'distances'], 
                all_obj_df.loc[all_obj_df['object'] == 2, 'goal'],
                [neuron_id], t_before=D_BEFORE_OBJ, t_after=D_AFTER_OBJ, bin_size=D_BIN_SIZE, ax=ax1,
                smoothing = D_SMOOTHING,
                pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}],
                errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}],
                raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}],
                eventline_kwargs={'lw': 0}, include_raster=True)
            this_y_max = ax1.get_ylim()[1]
            if this_y_max < 1:
                this_y_max = np.round(this_y_max + 0.05, 1)
                max_label = f'{this_y_max:.1f}'
            else:
                this_y_max = int(np.ceil(this_y_max))
                max_label = f'{this_y_max}'
            ax1.set(ylabel='Firing rate (spks/cm)', yticks=[0, this_y_max], yticklabels=['0', max_label],
                    xticks=[-D_BEFORE_OBJ, 0], xlabel='Distance from object entry (cm)')

            plt.tight_layout()
            
            # Save
            plt.savefig(join(path_dict['fig_path'], 'RemapNeurons', f'{region}',
                             f'{subject}_{date}_{probe}_neuron{neuron_id}.jpg'), dpi=300)
            plt.close(fig)

        
