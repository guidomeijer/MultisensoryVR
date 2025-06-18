# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:08:20 2023
@author: Guido Meijer

Event mapping
----------------------------

Surprise                1
Wheel A                 2
Wheel B                 3
Object 1 (house)        4
Object 2 (bridge)       5
Object 3 (desert)       6
Object 4 (playground)   7
Object 5 (nothing)      8
Sound 1 (rain)          9
Sound 2 (birds)         10
Camera                  11
Environment             12
Object 1 appear         13
Object 2 appear         14
Object 3 appear         15

"""

import os
from os.path import join, isdir, isfile
import numpy as np
import pandas as pd
from glob import glob
from scipy.ndimage import gaussian_filter1d
from itertools import chain
from logreader import create_bp_structure, compute_onsets, compute_offsets
from msvr_functions import paths

USE_SERVER = True

# Get paths
path_dict = paths(sync=USE_SERVER)
if USE_SERVER:
    search_folders = (join(path_dict['server_path'], 'Subjects'),
                      join(path_dict['local_data_path'], 'Subjects'))
else:
    search_folders = [join(path_dict['local_data_path'], 'Subjects')]

# Search for spikesort_me.flag
print('Looking for extract_me.flag..')
for root, directory, files in chain.from_iterable(os.walk(path) for path in search_folders):
    if 'ephys_session.flag' in files:
        print(f'\nFound extract_me.flag in {root}')

        data_files = glob(join(root, 'raw_behavior_data', '*.b64'))
        if len(data_files) == 0:
            print(f'No behavioral data found in {join(root, "raw_behavior_data")}')
            continue
        if len(data_files) > 1:
            print(f'Multiple behavioral log files found in {join(root, "raw_behavior_data")}')
            file_sizes = np.empty(len(data_files))
            for ii, this_file in enumerate(data_files):
                file_sizes[ii] = os.stat(this_file).st_size
            data_file = data_files[np.argmax(file_sizes)]
        elif len(data_files) == 1:
            data_file = data_files[0]            
        if isdir(join(root, 'raw_ephys_data')) and not isfile(join(root, 'raw_ephys_data', '_spikeglx_sync.times.npy')):
            print('Run ephys pipeline before behavioral extraction')
            continue

        # Unpack log file
        data = create_bp_structure(data_file)
        
        # Get timestamps in seconds relative to first timestamp
        time_s = (data['startTS'] - data['startTS'][0]) / 1000000
        
        # Unwind looped timestamps
        if np.where(np.diff(time_s) < 0)[0].shape[0] == 1:
            loop_point = np.where(np.diff(time_s) < 0)[0][0]
            time_s[loop_point+1:] = time_s[loop_point+1:] + time_s[loop_point]
        elif np.where(np.diff(time_s) < 0)[0].shape[0] > 1:
            print('Multiple time loop points detected! This is not supported yet.')
        
        # Get wheel distance (has to be floats for the smoothing step)
        wheel_distance = data['longVar'][:, 1].astype(float)

        # Calculate speed
        dist_filt = gaussian_filter1d(wheel_distance, 100)  # smooth wheel distance
        speed = np.abs(np.diff(dist_filt)) / np.diff(time_s)[0]

        # Convert back to int
        wheel_distance = wheel_distance.astype(int)

        # If this is an ephys session, synchronize timestamps with ephys
        if isdir(join(root, 'raw_ephys_data')):
            print('Ephys session detected, synchronizing timestamps with nidq')
            
            # Load in nidq sync pulses
            sync_times = np.load(join(root, 'raw_ephys_data', '_spikeglx_sync.times.npy'))
            sync_polarities = np.load(
                join(root, 'raw_ephys_data', '_spikeglx_sync.polarities.npy'))
            sync_channels = np.load(join(root, 'raw_ephys_data', '_spikeglx_sync.channels.npy'))
            nidq_pulses = sync_times[(sync_channels == 0) & (sync_polarities == 1)]

            # Load in Totalsync pulses
            totalsync_pulses = time_s[compute_onsets(data['digitalOut'][:, 4])]

            # Match the ephys and totalsync barcodes
            time_shift = np.nan
            for ii in range(nidq_pulses.shape[0]-20):
                
                if np.sum(np.abs(np.diff(nidq_pulses)[ii:ii+20] - np.diff(totalsync_pulses)[:20])) < 0.01:

                    # Ephys started before behavior
                    time_shift = nidq_pulses[ii] - totalsync_pulses[0]
                    new_totalsync = totalsync_pulses + time_shift
                    break

                if np.sum(np.abs(np.diff(totalsync_pulses)[ii:ii+20] - np.diff(nidq_pulses)[:20])) < 0.01:

                    # Behavior started before ephys
                    time_shift = -(totalsync_pulses[ii] - nidq_pulses[0])
                    new_totalsync = totalsync_pulses + time_shift
                    break
            if np.isnan(time_shift):
                print('No match found between ephys and totalsync barcodes!')
                continue

            # Ephys is the main clock so shift the totalsync timestamps accordingly
            time_s = time_s + time_shift
            
        # Check if there are trials
        if len(compute_onsets(data['digitalIn'][:, 12])) == 0:
            print('No trials found, deleting extraction flag')
            os.remove(join(root, 'extract_me.flag'))
            continue
            
        # For some reason sometimes the traces are inverted, invert back if necessary
        # If the trace is high for longer than low it's probably inverted
        first_half = data['digitalIn'].shape[0] // 2
        for jj in [1, 4, 5, 6, 7, 8, 13, 14, 15]:
            first_half_ch = data['digitalIn'][:first_half, jj]
            if np.sum(first_half_ch == 1) > np.sum(first_half_ch == 0):
                print(f'Channel {jj} inverted!')
                data['digitalIn'][:, jj] = 1 - data['digitalIn'][:, jj]
        
        # Now check whether the environment trace is inverted, objects should appear when the
        # environment trace is low (it's high in the tunnel and low in the env)
        first_half_env = data['digitalIn'][:first_half, 12]
        first_half_obj1 = data['digitalIn'][:first_half, 4]
        if np.sum(first_half_env[first_half_obj1 == 1]) / np.sum(first_half_obj1 == 1) > 0.1:
            print('Channel 12 (environment) inverted!')
            data['digitalIn'][:, 12] = 1 - data['digitalIn'][:, 12]
        
        # Sound traces should be zero in the tunnel, if not: invert them
        for jj in [9, 10]:  
            first_half_ch = data['digitalIn'][:first_half, jj]
            if np.sum(first_half_ch[first_half_env == 1]) / np.sum(first_half_env == 1) > 0.1:
                print(f'Channel {jj} inverted!')
                data['digitalIn'][:, jj] = 1 - data['digitalIn'][:, jj]
       
        # Extract trial onsets
        if compute_onsets(data['digitalIn'][:, 4])[0] < compute_offsets(data['digitalIn'][:, 12])[0]:
            # Missed the first environment TTL so first trial starts at 0 s
            env_start = np.concatenate(([np.nan], time_s[compute_offsets(data['digitalIn'][:, 12])]))
        else:
            env_start = time_s[compute_offsets(data['digitalIn'][:, 12])]

        # Extract reward times
        reward_times = time_s[compute_onsets(data['digitalOut'][:, 0])] 
       
        # Extract all event timings
        all_env_end = time_s[compute_onsets(data['digitalIn'][:, 12])]
        all_surprise = time_s[compute_onsets(data['digitalIn'][:, 1])]
        
        all_obj1_enter = time_s[compute_onsets(data['digitalIn'][:, 4])]
        all_obj2_enter = time_s[compute_onsets(data['digitalIn'][:, 5])]
        all_obj3_enter = time_s[compute_onsets(data['digitalIn'][:, 6])]
        all_obj4_enter = time_s[compute_onsets(data['digitalIn'][:, 7])]
        all_obj5_enter = time_s[compute_onsets(data['digitalIn'][:, 8])]
        
        all_obj1_exit = time_s[compute_offsets(data['digitalIn'][:, 4])]
        all_obj2_exit = time_s[compute_offsets(data['digitalIn'][:, 5])]
        all_obj3_exit = time_s[compute_offsets(data['digitalIn'][:, 6])]
        all_obj4_exit = time_s[compute_offsets(data['digitalIn'][:, 7])]
        all_obj5_exit = time_s[compute_offsets(data['digitalIn'][:, 8])]
        
        all_sound1_onsets = time_s[compute_onsets(data['digitalIn'][:, 9])]
        all_sound2_onsets = time_s[compute_onsets(data['digitalIn'][:, 10])]
        
        all_obj1_appear = time_s[compute_onsets(data['digitalIn'][:, 13])]
        all_obj2_appear = time_s[compute_onsets(data['digitalIn'][:, 14])]
        all_obj3_appear = time_s[compute_onsets(data['digitalIn'][:, 15])]
        
        # Only keep environment entries which have an object1 event and a trial end event
        discard_env_start = np.zeros(env_start.shape[0])
        for i, ts in enumerate(env_start[:-1]):
            if len(all_obj1_enter[(all_obj1_enter > ts)
                                  & (all_obj1_enter < env_start[i+1])]) == 0:
                discard_env_start[i] = 1
            if len(all_env_end[(all_env_end > ts)
                               & (all_env_end < env_start[i+1])]) == 0:
                discard_env_start[i] = 1
        env_start = env_start[~(discard_env_start).astype(bool)]

        # Pre-allocate trial arrays
        env_end = np.empty(env_start.shape[0]-1)
        surprise_time = np.empty(env_start.shape[0]-1)
        
        obj1_appear = np.empty(env_start.shape[0]-1)
        obj2_appear = np.empty(env_start.shape[0]-1)
        obj3_appear = np.empty(env_start.shape[0]-1)
        
        obj1_enter = np.empty(env_start.shape[0]-1)
        obj2_enter = np.empty(env_start.shape[0]-1)
        obj3_enter = np.empty(env_start.shape[0]-1)
        obj4_enter = np.empty(env_start.shape[0]-1)
        obj5_enter = np.empty(env_start.shape[0]-1)
        
        obj1_exit = np.empty(env_start.shape[0]-1)
        obj2_exit = np.empty(env_start.shape[0]-1)
        obj3_exit = np.empty(env_start.shape[0]-1)
        obj4_exit = np.empty(env_start.shape[0]-1)
        obj5_exit = np.empty(env_start.shape[0]-1)
        
        obj1_rewards = np.zeros(env_start.shape[0]-1).astype(int)
        obj2_rewards = np.zeros(env_start.shape[0]-1).astype(int)
        obj3_rewards = np.zeros(env_start.shape[0]-1).astype(int)
        
        obj1_position = np.zeros(env_start.shape[0]-1).astype(int)
        obj2_position = np.zeros(env_start.shape[0]-1).astype(int)
        obj3_position = np.zeros(env_start.shape[0]-1).astype(int)
        
        sound_onset = np.empty(env_start.shape[0]-1)
        sound_id = np.zeros(env_start.shape[0]-1).astype(int)

        # Loop over trials and get events per trial
        for i, ts in enumerate(env_start[:-1]):

            # Surprise event
            these_surprise = all_surprise[(all_surprise > ts)
                                          & (all_surprise < env_start[i+1])]
            if len(these_surprise) > 0:
                surprise_time[i] = these_surprise[0]
                        
            # Object enter and exit events
            these_obj1_enter = all_obj1_enter[(all_obj1_enter > ts)
                                              & (all_obj1_enter < env_start[i+1])]
            if len(these_obj1_enter) > 0:
                obj1_enter[i] = these_obj1_enter[0]
            these_obj2_enter = all_obj2_enter[(all_obj2_enter > ts)
                                              & (all_obj2_enter < env_start[i+1])]
            if len(these_obj2_enter) > 0:
                obj2_enter[i] = these_obj2_enter[0]
            these_obj3_enter = all_obj3_enter[(all_obj3_enter > ts)
                                              & (all_obj3_enter < env_start[i+1])]
            if len(these_obj3_enter) > 0:
                obj3_enter[i] = these_obj3_enter[0]
            these_obj4_enter = all_obj4_enter[(all_obj4_enter > ts)
                                              & (all_obj4_enter < env_start[i+1])]
            if len(these_obj4_enter) > 0:
                obj4_enter[i] = these_obj4_enter[0]
            these_obj5_enter = all_obj5_enter[(all_obj5_enter > ts)
                                              & (all_obj5_enter < env_start[i+1])]
            if len(these_obj5_enter) > 0:
                obj5_enter[i] = these_obj5_enter[0]
            
            these_obj1_exit = all_obj1_exit[(all_obj1_exit > ts) 
                                            & (all_obj1_exit < env_start[i+1])]
            if len(these_obj1_exit) > 0:
                obj1_exit[i] = these_obj1_exit[0]
            these_obj2_exit = all_obj2_exit[(all_obj2_exit > ts) 
                                            & (all_obj2_exit < env_start[i+1])]
            if len(these_obj2_exit) > 0:
                obj2_exit[i] = these_obj2_exit[0]
            these_obj3_exit = all_obj3_exit[(all_obj3_exit > ts)
                                            & (all_obj3_exit < env_start[i+1])]
            if len(these_obj3_exit) > 0:
                obj3_exit[i] = these_obj3_exit[0]   
            these_obj4_exit = all_obj4_exit[(all_obj4_exit > ts)
                                            & (all_obj4_exit < env_start[i+1])]
            if len(these_obj4_exit) > 0:
                obj4_exit[i] = these_obj4_exit[0]   
            these_obj5_exit = all_obj5_exit[(all_obj5_exit > ts)
                                            & (all_obj5_exit < env_start[i+1])]
            if len(these_obj5_exit) > 0:
                obj5_exit[i] = these_obj5_exit[0]   

            # Number of rewards given per object
            obj1_rewards[i] = np.sum((reward_times >= obj1_enter[i]-0.1)
                                     & (reward_times < obj1_exit[i]))
            obj2_rewards[i] = np.sum((reward_times >= obj2_enter[i]-0.1)
                                     & (reward_times < obj2_exit[i]))
            obj3_rewards[i] = np.sum((reward_times >= obj3_enter[i]-0.1)
                                     & (reward_times < obj3_exit[i]))

            # Which position was the object in
            obj1_position[i] = np.argsort([obj1_enter[i], obj2_enter[i], obj3_enter[i]])[0] + 1
            obj2_position[i] = np.argsort([obj1_enter[i], obj2_enter[i], obj3_enter[i]])[1] + 1
            obj3_position[i] = np.argsort([obj1_enter[i], obj2_enter[i], obj3_enter[i]])[2] + 1

            # Timestamp of appearance of object in first position
            these_obj1_appear = all_obj1_appear[(all_obj1_appear > ts)
                                                & (all_obj1_appear < env_start[i+1])]
            if len(these_obj1_appear) > 0:
                obj1_appear[i] = these_obj1_appear[0]
            these_obj2_appear = all_obj1_appear[(all_obj1_appear > ts)
                                                & (all_obj1_appear < env_start[i+1])]
            if len(these_obj2_appear) > 0:
                obj2_appear[i] = these_obj2_appear[0]
            these_obj3_appear = all_obj3_appear[(all_obj3_appear > ts)
                                                & (all_obj3_appear < env_start[i+1])]
            if len(these_obj3_appear) > 0:
                obj3_appear[i] = these_obj3_appear[0]

            # Sound on and offsets
            if all_sound1_onsets[(all_sound1_onsets > ts) & (all_sound1_onsets < env_start[i+1])].shape[0] > 0:
                sound_id[i] = 1
                sound_onset[i] = all_sound1_onsets[(all_sound1_onsets > ts) & (
                    all_sound1_onsets < env_start[i+1])][0]
            elif all_sound2_onsets[(all_sound2_onsets > ts) & (all_sound2_onsets < env_start[i+1])].shape[0] > 0:
                sound_id[i] = 2
                sound_onset[i] = all_sound2_onsets[(all_sound2_onsets > ts) & (
                    all_sound2_onsets < env_start[i+1])][0]

            # End of environment
            env_end[i] = all_env_end[(all_env_end > ts) & (all_env_end < env_start[i+1])][0]
        
        # Get camera timestamps
        camera_times = time_s[compute_onsets(data['digitalIn'][:, 11])]

        # Get lick times
        lick_times = time_s[compute_onsets(data['digitalOut'][:, 5])]
        
        # Get lick positions
        lick_pos = np.empty(lick_times.shape)
        for ii, lick_time in enumerate(lick_times):
            lick_pos[ii] = wheel_distance[np.argmin(np.abs(time_s - lick_time))]
        
        # Get breathing
        breathing = data['analog'][:, 6]
        
        # Get event positions
        lick_pos = wheel_distance[np.clip(
            np.searchsorted(time_s, lick_times, side='right') - 1, 0, wheel_distance.shape[0] - 1)]    
        reward_pos = wheel_distance[np.clip(
            np.searchsorted(time_s, reward_times, side='right') - 1, 0, wheel_distance.shape[0] - 1)]                 
        env_end_pos = wheel_distance[np.clip(
            np.searchsorted(time_s, env_end, side='right') - 1, 0, wheel_distance.shape[0] - 1)]
        obj1_enter_pos = wheel_distance[np.clip(
            np.searchsorted(time_s, obj1_enter, side='right') - 1, 0, wheel_distance.shape[0] - 1)]
        obj2_enter_pos = wheel_distance[np.clip(
            np.searchsorted(time_s, obj2_enter, side='right') - 1, 0, wheel_distance.shape[0] - 1)]
        obj3_enter_pos = wheel_distance[np.clip(
            np.searchsorted(time_s, obj3_enter, side='right') - 1, 0, wheel_distance.shape[0] - 1)]
        sound_onset_pos = wheel_distance[np.clip(
            np.searchsorted(time_s, sound_onset, side='right') - 1, 0, wheel_distance.shape[0] - 1)]
        env_start_pos = wheel_distance[np.clip(
            np.searchsorted(time_s, env_start[:-1], side='right') - 1, 0, wheel_distance.shape[0] - 1)]
        if np.isnan(env_start[0]):
            env_start_pos[0] = sound_onset_pos[0] - 25
        
        # Get spike positions
        probe_dirs = glob(join(root, 'probe0*'))
        if len(probe_dirs) > 0:
            for this_probe in probe_dirs:
                
                # Find for each spike its corresponding distance
                spike_times = np.load(join(this_probe, 'spikes.times.npy'))
                indices = np.searchsorted(time_s, spike_times, side='right') - 1
                indices = np.clip(indices, 0, wheel_distance.shape[0] - 1)
                spike_dist = wheel_distance[indices]
                np.save(join(this_probe, 'spikes.distances.npy'), spike_dist)
        
        # Save extracted events as ONE files
        np.save(join(root, 'continuous.wheelDistance.npy'), wheel_distance[:-1])
        np.save(join(root, 'continuous.wheelSpeed.npy'), speed)
        np.save(join(root, 'continuous.breathing.npy'), breathing[:-1])
        np.save(join(root, 'continuous.times.npy'), time_s[:-1])
        np.save(join(root, 'camera.times.npy'), camera_times)
        np.save(join(root, 'lick.times.npy'), lick_times)
        np.save(join(root, 'lick.positions.npy'), lick_pos)
        np.save(join(root, 'reward.times.npy'), reward_times)
        np.save(join(root, 'reward.positions.npy'), reward_pos)

        # Build trial dataframe
        trials = pd.DataFrame(data={
            'enterEnvTime': env_start[:-1], 'exitEnvTime': env_end,
            'enterEnvPos': env_start_pos, 'exitEnvPos': env_end_pos,
            'soundOnsetTime': sound_onset, 'soundOnsetPos': sound_onset_pos, 'soundId': sound_id,
            'appearObj1Time': obj1_appear,
            'enterObj1Time': obj1_enter, 'enterObj2Time': obj2_enter, 'enterObj3Time': obj3_enter,
            'enterObj1Pos': obj1_enter_pos, 'enterObj2Pos': obj2_enter_pos, 'enterObj3Pos': obj3_enter_pos,
            'exitObj1Time': obj1_exit, 'exitObj2Time': obj2_exit, 'exitObj3Time': obj3_exit,
            'rewardsObj1': obj1_rewards, 'rewardsObj2': obj2_rewards, 'rewardsObj3': obj3_rewards,
            'positionObj1': obj1_position, 'positionObj2': obj2_position, 'positionObj3': obj3_position
        })
        trials.to_csv(join(root, 'trials.csv'), index=False)

        # Delete extraction flag
        #os.remove(join(root, 'extract_me.flag'))
        if np.sum(np.isnan(obj1_enter)) > 0:
            print(f'{np.sum(np.isnan(obj1_enter))} missing enterObj1 events')
        if np.sum(np.isnan(obj2_enter)) > 0:
            print(f'{np.sum(np.isnan(obj2_enter))} missing enterObj2 events')
        if np.sum(np.isnan(obj3_enter)) > 0:
            print(f'{np.sum(np.isnan(obj3_enter))} missing enterObj3 events')
        if np.sum(np.isnan(obj1_exit)) > 0:
            print(f'{np.sum(np.isnan(obj1_exit))} missing exitObj1 events')
        if np.sum(np.isnan(obj2_exit)) > 0:
            print(f'{np.sum(np.isnan(obj2_exit))} missing exitObj2 events')
        if np.sum(np.isnan(obj3_exit)) > 0:
            print(f'{np.sum(np.isnan(obj3_exit))} missing exitObj3 events')
        print(f'Successfully extracted session with {trials.shape[0]} trials in {root}')
