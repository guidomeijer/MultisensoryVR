# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:08:20 2023

@author: Guido Meijer
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

USE_SERVER = False

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
    if 'extract_me.flag' in files:
        print(f'\nFound extract_me.flag in {root}')

        data_file = glob(join(root, 'raw_behavior_data', '*.b64'))
        if len(data_file) == 0:
            print(f'No behavioral data found in {join(root, "raw_behavior_data")}')
            continue
        if len(data_file) > 1:
            print(f'Multiple behavioral log files found in {join(root, "raw_behavior_data")}')
            continue
        if isdir(join(root, 'raw_ephys_data')) and not isfile(join(root, 'raw_ephys_data', '_spikeglx_sync.times.npy')):
            print('Run ephys pipeline before behavioral extraction')
            continue

        # Unpack log file
        data = create_bp_structure(data_file[0])
        
        # Get timestamps in seconds relative to first timestamp
        time_s = (data['startTS'] - data['startTS'][0]) / 1000000

        # Unwind looped timestamps
        if np.where(np.diff(time_s) < 0)[0].shape[0] == 1:
            loop_point = np.where(np.diff(time_s) < 0)[0][0]
            time_s[loop_point+1:] = time_s[loop_point+1:] + time_s[loop_point]
        elif np.where(np.diff(time_s) < 0)[0].shape[0] > 1:
            print('Multiple time loop points detected! This is not supported yet.')

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
            for ii in range(nidq_pulses.shape[0]):
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
            
        # Extract trial onsets
        if compute_onsets(data['digitalIn'][:, 8])[0] < compute_onsets(data['digitalIn'][:, 12])[0]:
            # Missed the first environment TTL so first trial starts at 0 s
            env_start = np.concatenate(([0], time_s[compute_offsets(data['digitalIn'][:, 12])]))
        else:
            env_start = time_s[compute_offsets(data['digitalIn'][:, 12])]

        # Extract reward times
        reward_times = time_s[compute_onsets(data['digitalOut'][:, 0])]

        # For some reason sometimes the traces are inverted, invert back if necessary
        for jj in range(8, 11):
            if np.sum(data['digitalIn'][:, jj] == 1) > np.sum(data['digitalIn'][:, jj] == 0):
                data['digitalIn'][:, jj] = 1 - data['digitalIn'][:, jj]

        # Extract all event timings
        all_env_end = time_s[compute_onsets(data['digitalIn'][:, 12])]
        all_first_obj_appear = time_s[compute_onsets(data['digitalIn'][:, 13])]
        all_reward_zones = time_s[compute_onsets(data['digitalIn'][:, 14])]
        all_obj1_enter = time_s[compute_onsets(data['digitalIn'][:, 8])]
        all_obj2_enter = time_s[compute_onsets(data['digitalIn'][:, 9])]
        all_obj3_enter = time_s[compute_onsets(data['digitalIn'][:, 10])]
        all_obj1_exit = time_s[compute_offsets(data['digitalIn'][:, 8])]
        all_obj2_exit = time_s[compute_offsets(data['digitalIn'][:, 9])]
        all_obj3_exit = time_s[compute_offsets(data['digitalIn'][:, 10])]
        all_sound0_onsets = time_s[compute_onsets(data['digitalIn'][:, 5])]
        all_sound1_onsets = time_s[compute_onsets(data['digitalIn'][:, 6])]
        all_sound2_onsets = time_s[compute_onsets(data['digitalIn'][:, 7])]
        all_sound0_offsets = time_s[compute_onsets(data['digitalIn'][:, 5])]
        all_sound1_offsets = time_s[compute_onsets(data['digitalIn'][:, 6])]
        all_sound2_offsets = time_s[compute_onsets(data['digitalIn'][:, 7])]
        
        # Only keep environment entries which have a first object appear event and a trial end event
        # TO DO: change to object appear
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
        first_obj_appear = np.empty(env_start.shape[0]-1)
        obj1_enter = np.empty(env_start.shape[0]-1)
        obj2_enter = np.empty(env_start.shape[0]-1)
        obj3_enter = np.empty(env_start.shape[0]-1)
        obj1_exit = np.empty(env_start.shape[0]-1)
        obj2_exit = np.empty(env_start.shape[0]-1)
        obj3_exit = np.empty(env_start.shape[0]-1)
        obj1_rewardzone = np.empty(env_start.shape[0]-1)
        obj2_rewardzone = np.empty(env_start.shape[0]-1)
        obj3_rewardzone = np.empty(env_start.shape[0]-1)
        obj1_rewards = np.zeros(env_start.shape[0]-1).astype(int)
        obj2_rewards = np.zeros(env_start.shape[0]-1).astype(int)
        obj3_rewards = np.zeros(env_start.shape[0]-1).astype(int)
        obj1_position = np.zeros(env_start.shape[0]-1).astype(int)
        obj2_position = np.zeros(env_start.shape[0]-1).astype(int)
        obj3_position = np.zeros(env_start.shape[0]-1).astype(int)
        sound_onset = np.empty(env_start.shape[0]-1)
        sound_offset = np.empty(env_start.shape[0]-1)
        sound_id = np.zeros(env_start.shape[0]-1).astype(int)

        # Loop over trials and get events per trial
        for i, ts in enumerate(env_start[:-1]):

            # Object enter and exit events
            these_obj1_enter = all_obj1_enter[(all_obj1_enter > ts) & (
                all_obj1_enter < env_start[i+1])]
            if len(these_obj1_enter) > 0:
                obj1_enter[i] = these_obj1_enter[0]
            else:
                obj1_enter[i] = np.nan
            these_obj2_enter = all_obj2_enter[(all_obj2_enter > ts) & (
                all_obj2_enter < env_start[i+1])]
            if len(these_obj2_enter) > 0:
                obj2_enter[i] = these_obj2_enter[0]
            else:
                obj2_enter[i] = np.nan
            these_obj3_enter = all_obj3_enter[(all_obj3_enter > ts) & (
                all_obj3_enter < env_start[i+1])]
            if len(these_obj3_enter) > 0:
                obj3_enter[i] = these_obj3_enter[0]
            else:
                obj3_enter[i] = np.nan
            these_obj1_exit = all_obj1_exit[(all_obj1_exit > ts) &
                                            (all_obj1_exit < env_start[i+1])]
            if len(these_obj1_exit) > 0:
                obj1_exit[i] = these_obj1_exit[0]
            else:
                obj1_exit[i] = np.nan
            these_obj2_exit = all_obj2_exit[(all_obj2_exit > ts) &
                                            (all_obj2_exit < env_start[i+1])]
            if len(these_obj2_exit) > 0:
                obj2_exit[i] = these_obj2_exit[0]
            else:
                obj2_exit[i] = np.nan
            these_obj3_exit = all_obj3_exit[(all_obj3_exit > ts) &
                                            (all_obj3_exit < env_start[i+1])]
            if len(these_obj3_exit) > 0:
                obj3_exit[i] = these_obj3_exit[0]
            else:
                obj3_exit[i] = np.nan
            these_obj1_rz = all_reward_zones[(all_reward_zones > obj1_enter[i]) &
                                             (all_reward_zones < obj1_exit[i])]
            if len(these_obj1_rz) > 0:
                obj1_rewardzone[i] = these_obj1_rz[0]
            else:
                obj1_rewardzone[i] = np.nan
            these_obj2_rz = all_reward_zones[(all_reward_zones > obj2_enter[i]) &
                                             (all_reward_zones < obj2_exit[i])]
            if len(these_obj2_rz) > 0:
                obj2_rewardzone[i] = these_obj2_rz[0]
            else:
                obj2_rewardzone[i] = np.nan
            these_obj3_rz = all_reward_zones[(all_reward_zones > obj3_enter[i]) &
                                             (all_reward_zones < obj3_exit[i])]
            if len(these_obj3_rz) > 0:
                obj3_rewardzone[i] = these_obj3_rz[0]
            else:
                obj3_rewardzone[i] = np.nan

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
            these_first_obj = all_first_obj_appear[(all_first_obj_appear > ts) & (
                all_first_obj_appear < env_start[i+1])]
            if len(these_first_obj) > 0:
                first_obj_appear[i] = these_first_obj[0]
            else:
                first_obj_appear[i] = np.nan

            # Sound on and offsets
            if all_sound0_onsets[(all_sound0_onsets > ts) & (all_sound0_onsets < env_start[i+1])].shape[0] > 0:
                sound_id[i] = 0
                sound_onset[i] = all_sound0_onsets[(all_sound0_onsets > ts) & (
                    all_sound0_onsets < env_start[i+1])][0]
                sound_offset[i] = all_sound0_offsets[(all_sound0_offsets > ts) & (
                    all_sound0_offsets < env_start[i+1])][0]
            elif all_sound1_onsets[(all_sound1_onsets > ts) & (all_sound1_onsets < env_start[i+1])].shape[0] > 0:
                sound_id[i] = 1
                sound_onset[i] = all_sound1_onsets[(all_sound1_onsets > ts) & (
                    all_sound1_onsets < env_start[i+1])][0]
                sound_offset[i] = all_sound1_offsets[(all_sound1_offsets > ts) & (
                    all_sound1_offsets < env_start[i+1])][0]
            elif all_sound2_onsets[(all_sound2_onsets > ts) & (all_sound2_onsets < env_start[i+1])].shape[0] > 0:
                sound_id[i] = 2
                sound_onset[i] = all_sound2_onsets[(all_sound2_onsets > ts) & (
                    all_sound2_onsets < env_start[i+1])][0]
                sound_offset[i] = all_sound2_offsets[(all_sound2_offsets > ts) & (
                    all_sound2_offsets < env_start[i+1])][0]

            # End of environment
            env_end[i] = all_env_end[(all_env_end > ts) & (all_env_end < env_start[i+1])][0]

        # Get camera timestamps
        camera_times = time_s[compute_onsets(data['digitalIn'][:, 11])]

        # Get wheel distance
        wheel_distance = data['longVar'][:, 1].astype(float)

        # Calculate speed
        dist_filt = gaussian_filter1d(wheel_distance, 100)  # smooth wheel distance
        speed = np.abs(np.diff(dist_filt)) / np.diff(time_s)[0]

        # Get lick times
        lick_times = time_s[compute_onsets(data['digitalOut'][:, 5])]
        
        # Get lick positions
        lick_pos = np.empty(lick_times.shape)
        for ii, lick_time in enumerate(lick_times):
            lick_pos[ii] = wheel_distance[np.argmin(np.abs(time_s - lick_time))]
        
        # Get breathing
        breathing = data['analog'][:, 6]
        
        # Get event positions
        print('Converting times into positions..')
        lick_pos = [wheel_distance[np.argmin(np.abs(time_s - i))] for i in lick_times]
        env_start_pos = [wheel_distance[np.argmin(np.abs(time_s - i))] for i in env_start[:-1]]
        env_end_pos = [wheel_distance[np.argmin(np.abs(time_s - i))] for i in env_end]
        obj1_enter_pos = [wheel_distance[np.argmin(np.abs(time_s - i))] for i in obj1_enter]
        obj2_enter_pos = [wheel_distance[np.argmin(np.abs(time_s - i))] for i in obj2_enter]
        obj3_enter_pos = [wheel_distance[np.argmin(np.abs(time_s - i))] for i in obj3_enter]
            
        # Save extracted events as ONE files
        np.save(join(root, 'continuous.wheelDistance.npy'), wheel_distance[:-1])
        np.save(join(root, 'continuous.wheelSpeed.npy'), speed)
        np.save(join(root, 'continuous.breathing.npy'), breathing[:-1])
        np.save(join(root, 'continuous.times.npy'), time_s[:-1])
        np.save(join(root, 'camera.times.npy'), camera_times)
        np.save(join(root, 'lick.times.npy'), lick_times)
        np.save(join(root, 'lick.positions.npy'), lick_pos)
        np.save(join(root, 'reward.times.npy'), reward_times)

        # Build trial dataframe
        trials = pd.DataFrame(data={
            'enterEnvTime': env_start[:-1], 'exitEnvTime': env_end,
            'enterEnvPos': env_start_pos, 'exitEnvPos': env_end_pos,
            'soundOnset': sound_onset, 'soundOffset': sound_offset, 'soundId': sound_id,
            'firstObjectAppear': first_obj_appear,
            'enterObj1': obj1_enter, 'enterObj2': obj2_enter, 'enterObj3': obj3_enter,
            'enterObj1Pos': obj1_enter_pos, 'enterObj2Pos': obj2_enter_pos, 'enterObj3Pos': obj3_enter_pos,
            'exitObj1': obj1_exit, 'exitObj2': obj2_exit, 'exitObj3': obj3_exit,
            'enterRewardZoneObj1': obj1_rewardzone, 'enterRewardZoneObj2': obj2_rewardzone,
            'enterRewardZoneObj3': obj3_rewardzone,
            'rewardsObj1': obj1_rewards, 'rewardsObj2': obj2_rewards, 'rewardsObj3': obj3_rewards,
            'positionObj1': obj1_position, 'positionObj2': obj2_position, 'positionObj3': obj3_position
        })
        trials.to_csv(join(root, 'trials.csv'), index=False)

        # Delete extraction flag
        os.remove(join(root, 'extract_me.flag'))
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
