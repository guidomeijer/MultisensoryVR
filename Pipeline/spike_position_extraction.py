# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:08:20 2023

@author: Guido Meijer
"""

from os import path
import numpy as np
import pandas as pd
from msvr_functions import paths

path_dict = paths(sync=False)
rec = pd.read_csv(path.join(path_dict['repo_path'], 'recordings.csv')).astype(str)


for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    probe_path = path.join(path_dict['local_data_path'], 'Subjects', subject, date, probe)
    if not path.isfile(path.join(probe_path, 'spikes.distances.npy')):
        
        print(f'Calculating spike distances for {subject} {date} {probe}..')
    
        # Load in wheel data
        wheel_dist = np.load(path.join(path.split(probe_path)[0], 'continuous.wheelDistance.npy'))
        wheel_times = np.load(path.join(path.split(probe_path)[0], 'continuous.times.npy'))
    
        # Load in spikes
        spike_times = np.load(path.join(probe_path, 'spikes.times.npy'))
    
        # Find for each spike its corresponding distance
        indices = np.searchsorted(wheel_times, spike_times, side='right') - 1
        indices = np.clip(indices, 0, wheel_dist.shape[0] - 1)
        spike_dist = wheel_dist[indices]
        
        # Save result
        np.save(path.join(probe_path, 'spikes.distances.npy'), spike_dist)
        
        
    if not path.isfile(path.join(probe_path, 'spikes.speeds.npy')):
        
        print(f'Calculating spike speeds for {subject} {date} {probe}..')
    
        # Load in wheel data
        wheel_speed = np.load(path.join(path.split(probe_path)[0], 'continuous.wheelSpeed.npy'))
        wheel_times = np.load(path.join(path.split(probe_path)[0], 'continuous.times.npy'))
    
        # Load in spikes
        spike_times = np.load(path.join(probe_path, 'spikes.times.npy'))
    
        # Find for each spike its corresponding distance
        indices = np.searchsorted(wheel_times, spike_times, side='right') - 1
        indices = np.clip(indices, 0, wheel_speed.shape[0] - 1)
        spike_speed = wheel_speed[indices]
        
        # Save result
        np.save(path.join(probe_path, 'spikes.speeds.npy'), spike_speed)
