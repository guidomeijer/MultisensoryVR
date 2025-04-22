# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:51:15 2025

By Guido Meijer
"""
from joblib import Parallel, delayed
from collections import defaultdict
import numpy as np
import pickle
import pandas as pd
from os.path import join
from brainbox.singlecell import calculate_peths
from msvr_functions import paths, load_neural_data, load_subjects

# Settings
MIN_SPEED = 50  # mm/s
CM_BEFORE = 10
CM_AFTER = 10
CM_BIN_SIZE = 1
CM_SMOOTHING = 2


def process_session(subject, date, probe, path_dict):
    print(f'\nStarting {subject} {date} {probe}..')
    
    try:
        session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
        spikes, clusters, channels = load_neural_data(session_path, probe)
        trials = pd.read_csv(join(session_path, 'trials.csv'))

        spikes_dist = spikes['distances'][spikes['speeds'] >= MIN_SPEED] / 10
        clusters_dist = spikes['clusters'][spikes['speeds'] >= MIN_SPEED]
        trials['enterEnvPos'] = trials['enterEnvPos'] / 10

        region_dict = defaultdict(lambda: None)

        for region in np.unique(clusters['region']):
            region_neurons = clusters['cluster_id'][clusters['region'] == region]
            
            context1, _ = calculate_peths(spikes_dist, clusters_dist, region_neurons,
                                          trials.loc[trials['soundId'] == 1, 'enterEnvPos'].values,
                                          CM_BEFORE, 150 + CM_AFTER, CM_BIN_SIZE, CM_SMOOTHING)
            context2, _ = calculate_peths(spikes_dist, clusters_dist, region_neurons,
                                          trials.loc[trials['soundId'] == 2, 'enterEnvPos'].values,
                                          CM_BEFORE, 150 + CM_AFTER, CM_BIN_SIZE, CM_SMOOTHING)
            envR = np.dstack((context1['means'], context2['means']))
            region_dict[region] = envR

        return region_dict

    except Exception as e:
        print(f"Error in {subject} {date} {probe}: {e}")
        return None
    
    
# Initialize
path_dict = paths()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# Run in parallel
results = Parallel(n_jobs=-1)(delayed(process_session)(subject, date, probe, path_dict)
                              for subject, date, probe in zip(rec['subject'], rec['date'], rec['probe']))

# Get distance bin ids
np.arange(-CM_BEFORE)

# Merge the results
env_act_dict = defaultdict(lambda: None)

for region_dict in results:
    if region_dict is None:
        continue
    for region, envR in region_dict.items():
        if env_act_dict[region] is None:
            env_act_dict[region] = envR
        else:
            env_act_dict[region] = np.vstack((env_act_dict[region], envR))

# Save results
with open(join(path_dict['save_path'], 'env_act_dict.pkl'), 'wb') as f:
    pickle.dump(dict(env_act_dict), f)
