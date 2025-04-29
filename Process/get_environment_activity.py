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
MIN_SPEED = 20  # mm/s
CM_BEFORE = 10
CM_AFTER = 10
CM_BIN_SIZE = 1
CM_SMOOTHING = 2
N_CORES = 10


def process_session(subject, date, probe, path_dict):    
    try:
        # Load data
        session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
        spikes, clusters, channels = load_neural_data(session_path, probe)
        trials = pd.read_csv(join(session_path, 'trials.csv'))
        
        # Get rewarded object positions
        rewarded_objects = np.array([np.sum(trials['rewardsObj1']) > 3,
                                     np.sum(trials['rewardsObj2']) > 3, 
                                     np.sum(trials['rewardsObj3']) > 3])
        obj_locations = np.array([trials.loc[0, 'positionObj1'],
                                  trials.loc[0, 'positionObj2'],
                                  trials.loc[0, 'positionObj3']])
        reward_locations = tuple(obj_locations[rewarded_objects])  # make it hashable

        # Apply speed threshold
        spikes_dist = spikes['distances'][spikes['speeds'] >= MIN_SPEED] / 10
        clusters_dist = spikes['clusters'][spikes['speeds'] >= MIN_SPEED]
        trials['enterEnvPos'] = trials['enterEnvPos'] / 10

        region_dict = defaultdict(lambda: None)

        for region in np.unique(clusters['region']):
            if region in ['root', 'ENT']:
                continue
            region_neurons = clusters['cluster_id'][clusters['region'] == region]
            
            context1, _ = calculate_peths(spikes_dist, clusters_dist, region_neurons,
                                          trials.loc[trials['soundId'] == 1, 'enterEnvPos'].values,
                                          CM_BEFORE, 150 + CM_AFTER, CM_BIN_SIZE, CM_SMOOTHING)
            context2, _ = calculate_peths(spikes_dist, clusters_dist, region_neurons,
                                          trials.loc[trials['soundId'] == 2, 'enterEnvPos'].values,
                                          CM_BEFORE, 150 + CM_AFTER, CM_BIN_SIZE, CM_SMOOTHING)
            envR = np.dstack((context1['means'], context2['means']))
            region_dict[region] = envR
        
        return reward_locations, region_dict

    except Exception as e:
        print(f"Error in {subject} {date} {probe}: {e}")
        return None

# Initialize
path_dict = paths()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# Run in parallel
results = Parallel(n_jobs=N_CORES)(
    delayed(process_session)(subject, date, probe, path_dict)
    for subject, date, probe in zip(rec['subject'], rec['date'], rec['probe'])
)

# Group results by reward_locations
split_env_dicts = defaultdict(lambda: defaultdict(lambda: None))

for result in results:
    if result is None:
        continue
    reward_locations, region_dict = result
    for region, envR in region_dict.items():
        if split_env_dicts[reward_locations][region] is None:
            split_env_dicts[reward_locations][region] = envR
        else:
            split_env_dicts[reward_locations][region] = np.vstack((split_env_dicts[reward_locations][region], envR))

# Add bin position to each and save
position_axis = np.arange(-CM_BEFORE, 150 + CM_AFTER, step=CM_BIN_SIZE) + (CM_BIN_SIZE / 2)

for reward_loc, env_act_dict in split_env_dicts.items():
    env_act_dict['position'] = position_axis
    filename = f"env_act_dict_{'_'.join(map(str, reward_loc))}.pkl"
    with open(join(path_dict['save_path'], filename), 'wb') as f:
        pickle.dump(dict(env_act_dict), f)