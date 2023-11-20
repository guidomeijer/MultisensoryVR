# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:39:26 2023

By Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from brainbox.singlecell import calculate_peths
from msvr_functions import paths, load_spikes
import cebra

# Settings
SUBJECT = '450409'
DATE = '20231012'
PROBE = 'probe00'

# Get paths
path_dict = paths()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters = load_spikes(session_path, PROBE)
wheel_time = np.load(join(path_dict['local_data_path'],
                     'Subjects', SUBJECT, DATE, 'continuous.times.npy'))
wheel_dist = np.load(join(path_dict['local_data_path'], 'Subjects',
                     SUBJECT, DATE, 'continuous.wheelDistance.npy'))

# Format data
peth, binned_spikes = calculate_peths(spikes['times'], spikes['clusters'], np.unique(spikes['clusters']),
                                      [np.min(wheel_time)], 0, np.max(spikes['times']),
                                      bin_size=0.05, smoothing=0)
binned_spikes = np.squeeze(binned_spikes)
binned_time = peth['tscale']
binned_loc = np.empty(binned_time.shape)
for i, this_time in enumerate(binned_time + np.min(wheel_time)):
    if np.mod(i, 1000) == 0:
        print(f'Timebin {i} of {binned_time.shape[0]}')
    binned_loc[i] = wheel_dist[np.argmin(np.abs(wheel_time - this_time))]
    
# Initialize CEBRA
cebra_posdir3_model = cebra.CEBRA(model_architecture='offset10-model',
                                  batch_size=512,
                                  learning_rate=3e-4,
                                  temperature=1,
                                  output_dimension=3,
                                  distance='cosine',
                                  conditional='time_delta',
                                  device='cpu',
                                  verbose=True,
                                  time_offsets=10)

# Fit model
cebra_posdir3_model.fit(binned_spikes.T, binned_loc)
cebra_posdir3 = cebra_posdir3_model.transform(binned_spikes.T)

# Plot result
fig = plt.figure(figsize=(5, 5))
ax = plt.subplot(projection='3d')
ax.scatter(cebra_posdir3[:,0],
            cebra_posdir3[:,1],
            cebra_posdir3[:,2],
            c=binned_loc,
            cmap='viridis', s=0.5)
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')


def plot_hippocampus(ax, embedding, gray = False, idx_order = (0,1,2)):
  

    r=ax.scatter(embedding [r_ind,idx1],
               embedding [r_ind,idx2],
               embedding [r_ind,idx3],
               c=r_c,
               cmap='viridis', s=0.5)
   
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    return ax