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
from sklearn.utils import shuffle
from brainbox.singlecell import calculate_peths
from msvr_functions import paths, load_neural_data, load_subjects

# Settings
MIN_SPEED = 20  # mm/s
CM_BEFORE = 10
CM_AFTER = 10
CM_BIN_SIZE = 1
CM_SMOOTHING = 2
N_CORES = 10
N_SHUFFLES = 500

# Initialize
path_dict = paths()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
position_axis = np.arange(-CM_BEFORE, 150 + CM_AFTER, step=CM_BIN_SIZE) + (CM_BIN_SIZE / 2)


# Function 
def get_shuffles(i, spikes_dist, clusters_dist, region_neurons, trials):
    trials_shuf = trials.copy()
    trials_shuf['soundId'] = shuffle(trials_shuf['soundId'].values, random_state=None)
    context1, _ = calculate_peths(spikes_dist, clusters_dist, region_neurons,
                                  trials_shuf.loc[trials_shuf['soundId'] == 1, 'enterEnvPos'].values,
                                  CM_BEFORE, 150 + CM_AFTER, CM_BIN_SIZE, CM_SMOOTHING)
    context2, _ = calculate_peths(spikes_dist, clusters_dist, region_neurons,
                                  trials_shuf.loc[trials_shuf['soundId'] == 2, 'enterEnvPos'].values,
                                  CM_BEFORE, 150 + CM_AFTER, CM_BIN_SIZE, CM_SMOOTHING)
    envR = np.dstack((context1['means'], context2['means']))
    return envR


# Loop over recordings
region_dict, region_dict_shuf, info_dict = dict(), dict(), dict()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    
    print(f'{subject} {date} {probe}.. [{i} of {rec.shape[0]}]')
    
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
    reward_locations = tuple(obj_locations[rewarded_objects])  
    if reward_locations != (1, 3):
        continue

    # Apply speed threshold
    spikes_dist = spikes['distances'][spikes['speeds'] >= MIN_SPEED] / 10
    clusters_dist = spikes['clusters'][spikes['speeds'] >= MIN_SPEED]
    trials['enterEnvPos'] = trials['enterEnvPos'] / 10

    for region in np.unique(clusters['region']):
        if region in ['root', 'ENT']:
            continue
        
        # Get neural activity over the environmet for the two contexts
        region_neurons = clusters['cluster_id'][clusters['region'] == region]
        context1, _ = calculate_peths(spikes_dist, clusters_dist, region_neurons,
                                      trials.loc[trials['soundId'] == 1, 'enterEnvPos'].values,
                                      CM_BEFORE, 150 + CM_AFTER, CM_BIN_SIZE, CM_SMOOTHING)
        context2, _ = calculate_peths(spikes_dist, clusters_dist, region_neurons,
                                      trials.loc[trials['soundId'] == 2, 'enterEnvPos'].values,
                                      CM_BEFORE, 150 + CM_AFTER, CM_BIN_SIZE, CM_SMOOTHING)
        envR = np.dstack((context1['means'], context2['means']))  # neurons x distance x context
        if region not in region_dict:
            region_dict[region] = envR
            info_dict[f'{region}_subject'] = [[subject] * envR.shape[0]]
            info_dict[f'{region}_date'] = [[date] * envR.shape[0]]
        else:
            region_dict[region] = np.vstack((region_dict[region], envR))
            info_dict[f'{region}_subject'].append([subject] * envR.shape[0])
            info_dict[f'{region}_date'].append([date] * envR.shape[0])
    
        # Run shuffles in parallel
        results = Parallel(n_jobs=N_CORES)(
            delayed(get_shuffles)(i, spikes_dist, clusters_dist, region_neurons, trials)
            for i in range(N_SHUFFLES))
        env_shuffles = np.stack([i for i in results], axis=3)  # neurons x dist x context x shuffles
        if region not in region_dict_shuf:
            region_dict_shuf[region] = env_shuffles
        else:
            region_dict_shuf[region] = np.vstack((region_dict_shuf[region], env_shuffles))
                
# Convert list to array
for key in info_dict.keys():
    info_dict[key] = np.concatenate(info_dict[key])

# Save to disk
info_dict['position'] = position_axis
with open(join(path_dict['local_data_path'], 'environment_activity_1-3.pkl'), 'wb') as f:
    pickle.dump(dict(region_dict), f)
with open(join(path_dict['local_data_path'], 'environment_activity_shuf_1-3.pkl'), 'wb') as f:
    pickle.dump(dict(region_dict_shuf), f)
with open(join(path_dict['local_data_path'], 'environment_info_1-3.pkl'), 'wb') as f:
    pickle.dump(dict(info_dict), f)