# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 10:45:57 2025

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, isdir, isfile
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from msvr_functions import (paths, peri_multiple_events_time_histogram, load_objects, load_trials,
                            load_neural_data, figure_style, load_subjects)
path_dict = paths(sync=False)
colors, dpi = figure_style()

# Settings
MIN_SPEED = 50  # mm/s

# Time
T_BEFORE = 2  # s
T_AFTER = 2
T_BIN_SIZE = 0.025
T_SMOOTHING = 0.1

# Distance
D_BEFORE_ENV = 10  # cm
D_AFTER_ENV = 160
D_BEFORE_OBJ = 30
D_AFTER_OBJ = 30
D_BIN_SIZE = 1
D_SMOOTHING = 2

# %% Plot object neuron in visual cortex
subject = '459601'
date = '20240411'
probe = 'probe00'
neuron_id = 579

session_path = path_dict['local_data_path'] / 'subjects' / subject / date
spikes, clusters, channels = load_neural_data(session_path, probe, histology=True, only_good=True)
trials = load_trials(subject, date)
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

# %% Plot
f, ax = plt.subplots(figsize=(2.5, 2), dpi=dpi)
peri_multiple_events_time_histogram(spikes_dist, clusters_dist,
                          trials['enterEnvPos'], np.ones(trials['enterEnvPos'].shape[0]),
                          [neuron_id], t_before=D_BEFORE_ENV, t_after=D_AFTER_ENV,
                          bin_size=D_BIN_SIZE, smoothing=D_SMOOTHING, ax=ax,
                          pethline_kwargs=[{'color': 'k', 'lw': 1}],
                          errbar_kwargs=[{'color': 'k', 'alpha': 0.3, 'lw': 0}],
                          raster_kwargs=[{'color': 'k', 'lw': 0.5}],
                          eventline_kwargs={'lw': 0}, include_raster=True)
ax.set(yticks=[0, 1], yticklabels=[0, 1], ylabel='Firing rate (spks/cm)',
       xlabel='Position (cm)')
ax.plot([45, 45], ax.get_ylim(), ls='--', color='grey', lw=0.75)
ax.plot([90, 90], ax.get_ylim(), ls='--', color='grey', lw=0.75)
ax.plot([135, 135], ax.get_ylim(), ls='--', color='grey', lw=0.75)
ax.yaxis.set_label_coords(-0.175, 0.75)

plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'Example neurons' / 'position_VIS.pdf')


# %% Expectation neuron

subject = '459601'
date = '20240411'
probe = 'probe00'
neuron_id = 234

session_path = path_dict['local_data_path'] / 'subjects' / subject / date
spikes, clusters, channels = load_neural_data(session_path, probe, histology=True, only_good=True)
trials = load_trials(subject, date)
all_obj_df = load_objects(subject, date)
all_obj_df.loc[all_obj_df['goal'] == 0, 'goal'] = 2
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
all_obj_df['distances'] = all_obj_df['distances'] / 10

# %% Plot
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4, 2), dpi=dpi, sharey=True)
        
peri_multiple_events_time_histogram(
    spikes_dist, clusters_dist, 
    all_obj_df.loc[all_obj_df['object'] == 1, 'distances'], 
    all_obj_df.loc[all_obj_df['object'] == 1, 'goal'],
    [neuron_id], t_before=D_BEFORE_OBJ, t_after=D_AFTER_OBJ, bin_size=D_BIN_SIZE, ax=ax1,
    smoothing = D_SMOOTHING, ylim=1,
    pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}],
    errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=True)
ax1.set(title='First object', ylabel='Firing rate (spks/cm)', yticks=[0, 1],
        xlabel='', xticks=[-D_BEFORE_OBJ, 0, D_AFTER_OBJ])
ax1.yaxis.set_label_coords(-0.3, 0.75)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
peri_multiple_events_time_histogram(
    spikes_dist, clusters_dist, 
    all_obj_df.loc[all_obj_df['object'] == 2, 'distances'], 
    all_obj_df.loc[all_obj_df['object'] == 2, 'goal'],
    [neuron_id], t_before=D_BEFORE_OBJ, t_after=D_AFTER_OBJ, bin_size=D_BIN_SIZE, ax=ax2,
    smoothing = D_SMOOTHING, ylim=1,
    pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}],
    errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=True)
ax2.set(title='Second object', yticks=[0, 1], ylabel='',
        xlabel='', xticks=[-D_BEFORE_OBJ, 0, D_AFTER_OBJ])
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
peri_multiple_events_time_histogram(
    spikes_dist, clusters_dist, 
    all_obj_df.loc[all_obj_df['object'] == 3, 'distances'], 
    np.ones(all_obj_df.loc[all_obj_df['object'] == 3].shape[0]),
    [neuron_id], t_before=D_BEFORE_OBJ, t_after=D_AFTER_OBJ, bin_size=D_BIN_SIZE, ax=ax3,
    smoothing = D_SMOOTHING, ylim=1,
    pethline_kwargs=[{'color': 'k', 'lw': 1}],
    errbar_kwargs=[{'color': 'k', 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': 'k', 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=True)
ax3.set(title='Control object', yticks=[0, 1], ylabel='',
        xlabel='', xticks=[-D_BEFORE_OBJ, 0, D_AFTER_OBJ])
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

f.text(0.5, 0.05, 'Distance from object entry (s)', ha='center')
plt.subplots_adjust(bottom=0.2, top=0.8)
plt.savefig(path_dict['google_drive_fig_path'] / 'Example neurons' / 'expectation_PERI36.pdf')

# %% outcome neuron

subject = '466395'
date = '20241113'
probe = 'probe00'
neuron_id = 566

session_path = path_dict['local_data_path'] / 'subjects' / subject / date
spikes, clusters, channels = load_neural_data(session_path, probe, histology=True, only_good=True)
trials = load_trials(subject, date)
all_obj_df = load_objects(subject, date)
all_obj_df.loc[all_obj_df['goal'] == 0, 'goal'] = 2
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
all_obj_df['distances'] = all_obj_df['distances'] / 10

# %% Plot
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(4, 2), dpi=dpi, sharey=True)
        
peri_multiple_events_time_histogram(
    spikes_dist, clusters_dist, 
    all_obj_df.loc[all_obj_df['object'] == 1, 'distances'], 
    all_obj_df.loc[all_obj_df['object'] == 1, 'goal'],
    [neuron_id], t_before=D_BEFORE_OBJ, t_after=D_AFTER_OBJ, bin_size=D_BIN_SIZE, ax=ax1,
    smoothing = D_SMOOTHING, ylim=1,
    pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}],
    errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=True)
ax1.set(title='First object', ylabel='Firing rate (spks/cm)', yticks=[0, 1],
        xlabel='', xticks=[-D_BEFORE_OBJ, 0, D_AFTER_OBJ])
ax1.yaxis.set_label_coords(-0.3, 0.75)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
peri_multiple_events_time_histogram(
    spikes_dist, clusters_dist, 
    all_obj_df.loc[all_obj_df['object'] == 2, 'distances'], 
    all_obj_df.loc[all_obj_df['object'] == 2, 'goal'],
    [neuron_id], t_before=D_BEFORE_OBJ, t_after=D_AFTER_OBJ, bin_size=D_BIN_SIZE, ax=ax2,
    smoothing = D_SMOOTHING, ylim=1,
    pethline_kwargs=[{'color': colors['goal'], 'lw': 1}, {'color': colors['no-goal'], 'lw': 1}],
    errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0}, {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': colors['goal'], 'lw': 0.5}, {'color': colors['no-goal'], 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=True)
ax2.set(title='Second object', yticks=[0, 1], ylabel='',
        xlabel='', xticks=[-D_BEFORE_OBJ, 0, D_AFTER_OBJ])
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
peri_multiple_events_time_histogram(
    spikes_dist, clusters_dist, 
    all_obj_df.loc[all_obj_df['object'] == 3, 'distances'], 
    np.ones(all_obj_df.loc[all_obj_df['object'] == 3].shape[0]),
    [neuron_id], t_before=D_BEFORE_OBJ, t_after=D_AFTER_OBJ, bin_size=D_BIN_SIZE, ax=ax3,
    smoothing = D_SMOOTHING, ylim=1,
    pethline_kwargs=[{'color': 'k', 'lw': 1}],
    errbar_kwargs=[{'color': 'k', 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': 'k', 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=True)
ax3.set(title='Control object', yticks=[0, 1], ylabel='',
        xlabel='', xticks=[-D_BEFORE_OBJ, 0, D_AFTER_OBJ])
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

f.text(0.5, 0.05, 'Distance from object entry (s)', ha='center')
plt.subplots_adjust(bottom=0.2, top=0.8)
plt.savefig(path_dict['google_drive_fig_path'] / 'Example neurons' / 'outcome_TEa.pdf')
