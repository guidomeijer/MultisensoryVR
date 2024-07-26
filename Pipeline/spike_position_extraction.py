# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:08:20 2023

@author: Guido Meijer
"""

import os
from os.path import join
import numpy as np
from glob import glob
from msvr_functions import paths

path_dict = paths(sync=False)

# Search for spikesort_me.flag
print('Looking for extract_me.flag..')
for root, directory, files in os.walk(path_dict['local_data_path']):
    if 'spikeposition_me.flag' in files:
        print(f'\nFound spikeposition_me.flag in {root}')

        # Load in wheel data
        wheel_dist = np.load(join(root, 'continuous.wheelDistance.npy'))
        wheel_times = np.load(join(root, 'continuous.times.npy'))

        probes = glob(join(root, 'probe*'))
        for p, this_probe in enumerate(probes):
            print(f'Starting probe {this_probe[-7:]}')

            # Load in spikes
            spike_times = np.load(join(this_probe, 'spikes.times.npy'))

            # Find for each spike its corresponding distance
            indices = np.searchsorted(wheel_times, spike_times, side='right') - 1
            indices = np.clip(indices, 0, wheel_dist.shape[0] - 1)
            spike_dist = wheel_dist[indices]
            
            # Save result
            np.save(join(this_probe, 'spikes.distances.npy'), spike_dist)
            print(f'Successfully extracted spike distances in {root}')

        # Remove flag
        os.remove(join(root, 'spikeposition_me.flag'))
