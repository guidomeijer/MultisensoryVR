# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:43:03 2023

By Guido Meijer
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import butter, filtfilt
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from brainbox import singlecell
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy.stats import pearsonr
import json, shutil, datetime
from glob import glob
from os.path import join, realpath, dirname, isfile, split, isdir
from pathlib import Path
from iblutil.numerical import ismember
from iblatlas.atlas import BrainRegions


def paths(sync=False, full_sync=False, force_sync=False):
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input

    This function also runs the synchronization between the server and the local data folder 
    once a day

    Input
    ------------------------
    sync : boolean
        When True data from the server will be synced with the local disk
    full_sync : boolean 
        When True also the raw data will be copied to the local drive
    force_sync : boolean    
        When True synchronization will be done regardless of how long ago the last sync was    

    Output
    ------------------------
    path_dict : dictionary
        Dict with the paths
    """

    # Get the paths
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        path_dict = dict()
        path_dict['fig_path'] = input('Path folder to save figures: ')
        path_dict['server_path'] = input('Path folder to server: ')
        path_dict['local_data_path'] = input('Path folder to local data: ')
        path_dict['save_path'] = join(dirname(realpath(__file__)), 'Data')
        path_dict['repo_path'] = dirname(realpath(__file__))
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(path_dict, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        path_dict = json.load(json_file)
        path_dict = {key: Path(value) for key, value in path_dict.items()}
        

    # Synchronize data from the server with the local data folder

    # Create Subjects folder if it doesn't exist
    if not isdir(join(path_dict['local_data_path'], 'Subjects')):
        os.mkdir(join(path_dict['local_data_path'], 'Subjects'))

    # Read in the time of last sync
    if isfile(join(path_dict['local_data_path'], 'sync_timestamp.txt')):
        f = open(join(path_dict['local_data_path'], 'sync_timestamp.txt'), 'r')
        sync_time = datetime.datetime.strptime(f.read(), '%Y%m%d_%H%M')
        f.close()
    else:
        # If never been synched set date to yesterday so that it runs the first time
        sync_time = datetime.datetime.now() - datetime.timedelta(hours=24)

    # Synchronize server with local data once a day
    if force_sync:
        sync = True
    if sync:
        if ((datetime.datetime.now() - sync_time).total_seconds() > 12*60*60) | force_sync:
            print('Synchronizing data from server with local data folder')
            if not isdir(path_dict['server_path']):
                print('Server path not found')
                return path_dict
            
            # Loop over subjects
            subjects = os.listdir(join(path_dict['server_path'], 'Subjects'))
            for i, subject in enumerate(subjects):
                if not isdir(join(path_dict['local_data_path'], 'Subjects', subject)):
                    os.mkdir(join(path_dict['local_data_path'], 'Subjects', subject))
                    
                # Loop over sessions
                sessions = os.listdir(join(path_dict['server_path'], 'Subjects', subject))
                for j, session in enumerate(sessions):
                    
                    # Get server path and files on server
                    server_path = join(path_dict['server_path'], 'Subjects', subject, session)
                    server_files = [f for f in os.listdir(server_path)
                                    if (isfile(join(server_path, f)) & (f[-4:] != 'flag'))]
                    if len(server_files) == 0:
                        continue
                    
                    # Get local path and files on local disk
                    local_path = join(path_dict['local_data_path'], 'Subjects', subject, session)
                    if not isdir(local_path):
                        os.mkdir(local_path)
                    local_files = [f for f in os.listdir(local_path)
                                   if isfile(join(local_path, f)) & (f[-4:] != 'flag')]    
                    
                    # Copy files one way or another depending on which has the most files
                    if len(server_files) > len(local_files):
                        print(f'Copying files from server {server_path}')
                        for f, file in enumerate(server_files):
                            if not isfile(join(local_path, file)):
                                shutil.copyfile(join(server_path, file), join(local_path, file))
                    elif len(server_files) < len(local_files):
                        print(f'Copying files to server {server_path}')
                        for f, file in enumerate(local_files):
                            if not isfile(join(server_path, file)):
                                shutil.copyfile(join(local_path, file), join(server_path, file))
                    
                    # Synchronize neural data 
                    if isdir(join(local_path, 'probe00')):
                        local_probe_files = [f for f in os.listdir(join(local_path, 'probe00'))
                                             if isfile(join(local_path, f))]
                        server_probe_files = [f for f in os.listdir(join(server_path, 'probe00'))
                                              if isfile(join(server_path, f))]   
                        if len(local_probe_files) > len(server_probe_files):
                            print(f'{subject} {session} | Copying probe00 files to server {server_path}')
                            for f, file in enumerate(local_probe_files):
                                if not isfile(join(server_path, 'probe00', file)):
                                    shutil.copyfile(join(local_path, 'probe00', file),
                                                    join(server_path, 'probe00', file))
                        elif len(local_probe_files) < len(server_probe_files):
                            print(f'{subject} {session} | Copying probe00 files from server {server_path}')
                            for f, file in enumerate(server_probe_files):
                                if not isfile(join(local_path, 'probe00', file)):
                                    shutil.copyfile(join(server_path, 'probe00', file),
                                                    join(local_path, 'probe00', file))
                    if isdir(join(local_path, 'probe01')):
                        local_probe_files = [f for f in os.listdir(join(local_path, 'probe01'))
                                             if isfile(join(local_path, f))]
                        server_probe_files = [f for f in os.listdir(join(server_path, 'probe01'))
                                              if isfile(join(server_path, f))]   
                        if len(local_probe_files) > len(server_probe_files):
                            print(f'{subject} {session} | Copying probe01 files to server {server_path}')
                            for f, file in enumerate(local_probe_files):
                                if not isfile(join(server_path, 'probe01', file)):
                                    shutil.copyfile(join(local_path, 'probe01', file),
                                                    join(server_path, 'probe01', file))
                        elif len(local_probe_files) < len(server_probe_files):
                            print(f'{subject} {session} | Copying probe01 files from server {server_path}')
                            for f, file in enumerate(server_probe_files):
                                if not isfile(join(local_path, 'probe01', file)):
                                    shutil.copyfile(join(server_path, 'probe01', file),
                                                    join(local_path, 'probe01', file))
                        
                    # If the probe folder does not exist on local drive, copy it from server
                    if (isdir(join(server_path, 'probe00'))) & (~isdir(join(local_path, 'probe00'))):
                        print(f'{subject} {session} | Copying probe00 folder from server')
                        if full_sync:
                            shutil.copytree(join(server_path, 'probe00'), join(local_path, 'probe00'))
                        else:
                            os.mkdir(join(local_path, 'probe00'))
                            server_probe_files = [f for f in os.listdir(join(server_path, 'probe00'))
                                                  if isfile(join(server_path, 'probe00', f))]   
                            for f, file in enumerate(server_probe_files):
                                shutil.copyfile(join(server_path, 'probe00', file),
                                                join(local_path, 'probe00', file))
                                
                    if (isdir(join(server_path, 'probe01'))) & (~isdir(join(local_path, 'probe01'))):
                        print(f'{subject} {session} | Copying probe01 folder from server')
                        if full_sync:
                            shutil.copytree(join(server_path, 'probe01'), join(local_path, 'probe01'))
                        else:
                            os.mkdir(join(local_path, 'probe01'))
                            server_probe_files = [f for f in os.listdir(join(server_path, 'probe01'))
                                                  if isfile(join(server_path, 'probe01', f))]   
                            for f, file in enumerate(server_probe_files):
                                shutil.copyfile(join(server_path, 'probe01', file),
                                                join(local_path, 'probe01', file))
                    
                    # Copy raw data from server if a full sync is requested
                    if (not isdir(join(server_path, 'raw_video_data'))) & full_sync:
                        print(
                            f'Copying raw video data {join(path_dict["server_path"], "Subjects", subject, session)}')
                        shutil.copytree(join(server_path, 'raw_video_data'),
                                        join(local_path, 'raw_video_data'))
                    if isdir(join(path_dict['server_path'], 'Subjects', subject, session, 'raw_ephys_data')) & full_sync:
                        if not isdir(join(local_path, 'raw_ephys_data')):
                            print(
                                f'Copying raw ephys data {join(path_dict["server_path"], "Subjects", subject, session)}')
                            shutil.copytree(join(server_path, 'raw_ephys_data'),
                                            join(local_path, 'raw_ephys_data'))
                    if (not isdir(join(local_path, 'raw_behavior_data'))) & full_sync:
                        print(
                            f'Copying raw behavior data {join(path_dict["server_path"], "Subjects", subject, session)}')
                        shutil.copytree(join(server_path, 'raw_behavior_data'),
                                        join(local_path, 'raw_behavior_data'))

            # Update synchronization timestamp
            with open(join(path_dict['local_data_path'], 'sync_timestamp.txt'), 'w') as f:
                f.write(datetime.datetime.now().strftime('%Y%m%d_%H%M'))
                f.close()
            print('Done')

    return path_dict


def load_trials(subject, date):
    path_dict = paths()
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    trials = trials[trials['soundId'] != 0]
    return trials


def load_objects(subject, date):
    """
    Parameters
    ----------
    subject : str
    date : str
    reorder : boolean, optional
        Whether to reorder the object id's such that:
            object 1: the first encountered rewarded object
            object 2: the second encountered rewarded object
            object 3: the control object (wherever it is) 
            The default is True.

    Returns
    -------
    all_obj_df : pandas dataframe
    """
    
    # Initialize
    path_dict = paths()
    subjects = load_subjects()
    
    # Load in trials
    trials = load_trials(subject, date)
    
    # Get reward contingencies
    sound1_obj = subjects.loc[subjects['SubjectID'] == subject, 'Sound1Obj'].values[0]
    sound2_obj = subjects.loc[subjects['SubjectID'] == subject, 'Sound2Obj'].values[0]
    control_obj = subjects.loc[subjects['SubjectID'] == subject, 'ControlObject'].values[0]

    # Reorder rewarded objects such that object 1 is the first encountered rewarded object
    sound1_obj_pos = trials[f'positionObj{sound1_obj}'].values[0]
    sound2_obj_pos = trials[f'positionObj{sound2_obj}'].values[0]
    if sound1_obj_pos < sound2_obj_pos:
        sound1_obj_id = 1
        sound2_obj_id = 2
    else:
        sound1_obj_id = 2
        sound2_obj_id = 1        

    # Object id mapping
    obj_mapping = {1: 'house', 2: 'bridge', 3: 'desert'}

    # Prepare trial data
    rew_obj1_df = pd.DataFrame(data={'times': trials[f'enterObj{sound1_obj}Time'],
                                     'distances': trials[f'enterObj{sound1_obj}Pos'],
                                     'object': sound1_obj_id, 'sound': trials['soundId'],
                                     'goal': (trials['soundId'] == 1).astype(int),
                                     'rewarded': trials[f'rewardsObj{sound1_obj}'],
                                     'object_appearance': obj_mapping[sound1_obj],
                                     'trial_nr': trials.index.values})
    rew_obj2_df = pd.DataFrame(data={'times': trials[f'enterObj{sound2_obj}Time'],
                                     'distances': trials[f'enterObj{sound2_obj}Pos'],
                                     'object': sound2_obj_id, 'sound': trials['soundId'],
                                     'goal': (trials['soundId'] == 2).astype(int),
                                     'rewarded': trials[f'rewardsObj{sound2_obj}'],
                                     'object_appearance': obj_mapping[sound2_obj],
                                     'trial_nr': trials.index.values})
    control_obj_df = pd.DataFrame(data={'times': trials[f'enterObj{control_obj}Time'],
                                        'distances': trials[f'enterObj{control_obj}Pos'],
                                        'object': 3, 'sound': trials['soundId'],
                                        'goal': 0, 'rewarded': trials[f'rewardsObj{control_obj}'],
                                        'object_appearance': obj_mapping[control_obj],
                                        'trial_nr': trials.index.values})
    
    # Add reward times per object
    reward_times = np.load(join(path_dict['local_data_path'], 'Subjects', subject, date, 'reward.times.npy'))
    rew_obj1_df.loc[rew_obj1_df['rewarded'] > 0, 'reward_times'] = [
        reward_times[np.argmin(np.abs(i - reward_times))]
        for i in rew_obj1_df.loc[rew_obj1_df['rewarded'] > 0, 'times']]
    rew_obj2_df.loc[rew_obj2_df['rewarded'] > 0, 'reward_times'] = [
        reward_times[np.argmin(np.abs(i - reward_times))]
        for i in rew_obj2_df.loc[rew_obj2_df['rewarded'] > 0, 'times']]
    
    # Create dataframe
    all_obj_df = pd.concat((rew_obj1_df, rew_obj2_df, control_obj_df))
    all_obj_df = all_obj_df.sort_values(by='times').reset_index(drop=True)
    
    return all_obj_df


def figure_style(font_size=7):
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": font_size,
                "figure.titlesize": font_size,
                "figure.labelweight": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                'legend.fontsize': font_size,
                'legend.title_fontsize': font_size,
                'legend.frameon': False,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['backend'] = 'QtAgg'

    colors = {
        'obj1': sns.color_palette('Set2')[0],
        'obj2': sns.color_palette('Set2')[1],
        'obj3': sns.color_palette('Set2')[2],
        'goal': matplotlib.colors.to_rgb('mediumseagreen'),
        'no-goal': matplotlib.colors.to_rgb('tomato'),
        'control': matplotlib.colors.to_rgb('gray'),
        'sound1': matplotlib.colors.to_rgb('goldenrod'),
        'sound2': matplotlib.colors.to_rgb('darkorchid'),
        'PERI 36': sns.color_palette('Set3')[5],
        'PERI 35': sns.color_palette('Set3')[3],
        'sPERI': sns.color_palette('Set3')[5],
        'dPERI': sns.color_palette('Set3')[3],
        'TEa': sns.color_palette('Set3')[2],
        'VIS': sns.color_palette('Set3')[0],
        'AUD': sns.color_palette('Set3')[6],
        'HPC': sns.color_palette('Set3')[4],
        'CA1': sns.color_palette('Set3')[4],
        'vCA1': sns.color_palette('Set3')[4],
        'dCA1': sns.color_palette('Dark2')[4],
        'TBD': sns.color_palette('tab10')[9],
        'ENT': sns.color_palette('Dark2')[7],
        'TBD': sns.color_palette('Dark2')[0],
        'TBD': sns.color_palette('Accent')[0],
        'TBD': sns.color_palette('Accent')[1],
        'TBD': sns.color_palette('Accent')[2],
        'TBD': sns.color_palette('tab10')[8],
        }

    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 4
    if dpi > 300:
        dpi = 300
    return colors, dpi


def load_subjects():
    path_dict = paths(sync=False)
    subjects = pd.read_csv(join(path_dict['repo_path'], 'subjects.csv'),
                           delimiter=';|,',
                           engine='python')
    subjects['SubjectID'] = subjects['SubjectID'].astype(str)
    subjects['DateFinalTask'] = subjects['DateFinalTask'].astype(str)
    subjects['DateFinalTask'] = [datetime.datetime.strptime(i, '%Y%m%d').date() for i
                                 in subjects['DateFinalTask']]
    return subjects


def load_neural_data(session_path, probe, histology=True, only_good=True, min_fr=0.1):
    """
    Helper function to read in the spike sorting output from the Power Pixels pipeline.

    Parameters
    ----------
    session_path : str
        Full path to the top-level folder of the session.
    probe : str
        Name of the probe to load in.
    histology : bool, optional
        Whether to load the channel location and brain regions from the output of the alignment GUI.
        If False, no brain regions will be provided. The default is True.
    only_good : bool, optional
        Whether to only load in neurons that are labelled 'good' by automatic curation, either 
        IBL labelled good or machine learning lablled good. The default is True.
    min_fr : float, optional
        Only return neurons with a firing rate of minimally this value in spikes/s over the entire 
        recording

    Returns
    -------
    spikes : dict
        A dictionary containing data per spike
    clusters : dict
        A dictionary containing data per cluster (i.e. neuron)
    channels : dict
        A dictionary containing data per channel 
    """
    
    # Load in spiking data
    spikes = dict()
    spikes['times'] = np.load(join(session_path, probe, 'spikes.times.npy'))
    spikes['clusters'] = np.load(join(session_path, probe, 'spikes.clusters.npy'))
    spikes['amps'] = np.load(join(session_path, probe, 'spikes.amps.npy'))
    spikes['depths'] = np.load(join(session_path, probe, 'spikes.depths.npy'))
    spikes['samples'] = np.load(join(session_path, probe, 'spikes.samples.npy'))
    if isfile(join(session_path, probe, 'spikes.distances.npy')):
        spikes['distances'] = np.load(join(session_path, probe, 'spikes.distances.npy'))
    if isfile(join(session_path, probe, 'spikes.speeds.npy')):
        spikes['speeds'] = np.load(join(session_path, probe, 'spikes.speeds.npy'))
        
    # Load in cluster data
    clusters = dict()
    clusters['channels'] = np.load(join(session_path, probe, 'clusters.channels.npy'))
    clusters['depths'] = np.load(join(session_path, probe, 'clusters.depths.npy'))
    clusters['amps'] = np.load(join(session_path, probe, 'clusters.amps.npy'))
    clusters['cluster_id'] = np.arange(clusters['channels'].shape[0])
    
    # Add cluster qc metrics
    clusters['ibl_label'] = np.load(join(session_path, probe, 'clusters.IBLLabel.npy'))
    clusters['ml_label'] = np.load(join(session_path, probe, 'clusters.MLLabel.npy'))
    if isfile(join(session_path, probe, 'clusters.manualLabels.npy')):
        clusters['manual_label'] = np.load(join(session_path, probe, 'clusters.manualLabels.npy'))
        
    # Add neuron firing rates
    if isfile(join(session_path, probe, 'clusters.firingRates.npy')):
        clusters['firing_rate'] = np.load(join(session_path, probe, 'clusters.firingRates.npy'))
    else:
        clusters['firing_rate'] =  np.array([spikes['times'][spikes['clusters'] == i].shape[0]
                                             / spikes['times'][-1] for i in clusters['cluster_id']])
        np.save(join(session_path, probe, 'clusters.firingRates.npy'), clusters['firing_rate'])
    
    # Load in channel data
    channels = dict()
    if histology:
        if isfile(join(session_path, probe, 'channels.brainLocations.csv')):
            channels_df = pd.read_csv(join(session_path, probe, 'channels.brainLocations.csv'))
            channels = {col: np.array(channels_df[col]) for col in channels_df.columns}
        else:
            
            channel_loc_files = glob(join(session_path, probe, 'channel_locations*'))
            if len(channel_loc_files) == 0:
                raise Exception('No aligned channel locations found! Set histology to False to load data without brain regions.')
            elif len(channel_loc_files) == 1:
                
                # One shank recording
                f = open(join(session_path, probe, 'channel_locations.json'))
                channel_locations = json.load(f)
                f.close()
                channel_locations_df = pd.DataFrame(data=channel_locations).transpose()
                
            elif len(channel_loc_files) == 4:
                
                # Four shank recording
                channel_locations_df = pd.DataFrame()
                for this_file in channel_loc_files:
                    f = open(this_file)
                    channel_locations = json.load(f)
                    f.close()
                    this_df = pd.DataFrame(data=channel_locations).transpose()
                    channel_locations_df = pd.concat((channel_locations_df, this_df))
                    
            # Match xyz channel location and brain region to channels by local coordinates
            local_coordinates = np.load(join(session_path, probe, 'channels.localCoordinates.npy'))
            channels_df = pd.DataFrame(index=np.arange(local_coordinates.shape[0]),
                                       columns=['acronym', 'atlas_id', 'x', 'y', 'z'])
            for i, (lateral_um, axial_um) in enumerate(zip(local_coordinates[:, 0], local_coordinates[:, 1])):
                this_ch = ((channel_locations_df['lateral'] == lateral_um)
                           & (channel_locations_df['axial'] == axial_um)).values
                channels_df.loc[i, 'acronym'] = channel_locations_df.loc[this_ch, 'brain_region'].values[0]
                channels_df.loc[i, 'atlas_id'] = channel_locations_df.loc[this_ch, 'brain_region_id'].values[0]
                channels_df.loc[i, 'x'] = channel_locations_df.loc[this_ch, 'x'].values[0]
                channels_df.loc[i, 'y'] = channel_locations_df.loc[this_ch, 'y'].values[0]
                channels_df.loc[i, 'z'] = channel_locations_df.loc[this_ch, 'z'].values[0]
            channels_df['lateral_um'] = local_coordinates[:, 0]
            channels_df['axial_um'] = local_coordinates[:, 1]
            
            # Save to disk
            channels_df.to_csv(join(session_path, probe, 'channels.brainLocation.csv'), index=False)
            
            # Change into dict
            channels = {col: np.array(channels_df[col]) for col in channels_df.columns}            
        
        # Use the channel location to infer the locations of the neurons
        clusters['x'] = channels['x'][clusters['channels']]
        clusters['y'] = channels['y'][clusters['channels']]
        clusters['z'] = channels['z'][clusters['channels']]
        clusters['acronym'] = channels['acronym'][clusters['channels']]
        clusters['region'] = combine_regions(clusters['acronym'], abbreviate=True, brainregions=BrainRegions())
        clusters['region'][(clusters['region'] == 'CA1') & (clusters['z'] < -2000)] = 'vCA1'
        clusters['region'][(clusters['region'] == 'CA1') & (clusters['z'] > -2000)] = 'dCA1'
        clusters['full_region'] = combine_regions(clusters['acronym'], abbreviate=False, 
                                                  brainregions=BrainRegions())
        clusters['full_region'][(clusters['full_region'] == 'CA1') & (clusters['z'] < -2000)] = 'ventral CA1'
        clusters['full_region'][(clusters['full_region'] == 'CA1') & (clusters['z'] > -2000)] = 'dorsal CA1'
        
    # Exclude neurons that are not labelled good or with firing rates which are too low
    select_units = np.ones(clusters['cluster_id'].shape[0]).astype(bool)
    select_units[np.where(clusters['firing_rate'] < min_fr)[0]] = False
    if only_good:     
        select_units[np.where((clusters['ml_label'] == 0) & (clusters['ibl_label'] < 1))[0]] = False
    keep_units = clusters['cluster_id'][select_units]
    spikes['times'] = spikes['times'][np.isin(spikes['clusters'], keep_units)]
    spikes['amps'] = spikes['amps'][np.isin(spikes['clusters'], keep_units)]
    spikes['depths'] = spikes['depths'][np.isin(spikes['clusters'], keep_units)]
    if 'distances' in spikes.keys():
        spikes['distances'] = spikes['distances'][np.isin(spikes['clusters'], keep_units)]
    if 'speeds' in spikes.keys():
        spikes['speeds'] = spikes['speeds'][np.isin(spikes['clusters'], keep_units)]
    spikes['clusters'] = spikes['clusters'][np.isin(spikes['clusters'], keep_units)]
    if histology:
        clusters['x'] = clusters['x'][keep_units]
        clusters['y'] = clusters['y'][keep_units]
        clusters['z'] = clusters['z'][keep_units]
        clusters['acronym'] = clusters['acronym'][keep_units]
        clusters['region'] = clusters['region'][keep_units]
        clusters['full_region'] = clusters['full_region'][keep_units]
    clusters['depths'] = clusters['depths'][keep_units]
    clusters['amps'] = clusters['amps'][keep_units]
    clusters['firing_rate'] = clusters['firing_rate'][keep_units]
    clusters['cluster_id'] = clusters['cluster_id'][keep_units]
    clusters['ml_label'] = clusters['ml_label'][keep_units]
    clusters['ibl_label'] = clusters['ibl_label'][keep_units]
    
    return spikes, clusters, channels
    

def load_multiple_probes(session_path, probes=[], **kwargs):
    """
    Load all simultaneously recorded probes of a single session

    Parameters
    ----------
    session_path : str
        Path to session data.
    probes : list
        List of probes to load in, by default load all probes
    **kwargs 
        Extra inputs to the load_neural_data function.

    Returns
    -------
    spikes : dict
    clusters : dict
    channels : dict
    """
    
    if len(probes) == 0:
        # Get all probes of session
        probe_paths = glob(join(session_path, 'probe*'))
        probes = [split(i)[-1] for i in probe_paths]
    
    # Load in neural data per probe
    spikes, clusters, channels = dict(), dict(), dict()
    for probe in probes:
        spikes[probe], clusters[probe], channels[probe] = load_neural_data(
            session_path, probe, **kwargs)
    
    return spikes, clusters, channels    
    

def remap(acronyms, source='Allen', dest='Beryl', brainregions=None):
    """
    Remaps a list of brain region acronyms from one mapping source to another.
    Parameters:
        acronyms (list or array-like): A list of brain region acronyms to be remapped.
        source (str, optional): The source mapping to use for remapping. Default is 'Allen'.
        dest (str, optional): The destination mapping to remap to. Default is 'Beryl'.
        brainregions (BrainRegions, optional): An instance of the BrainRegions class. 
            If not provided, a new instance will be created.
    Returns:
        list: A list of remapped brain region acronyms corresponding to the destination mapping.
    Notes:
        - The function uses the `BrainRegions` class to handle mappings and acronym conversions.
        - The `ismember` function is used to find indices of the acronyms in the source mapping.
        - Ensure that the `BrainRegions` class and its methods (`acronym2id`, `get`, etc.) 
          are properly implemented and accessible.
    """

    br = brainregions or BrainRegions()
    _, inds = ismember(br.acronym2id(acronyms), br.id[br.mappings[source]])
    remapped_acronyms = br.get(br.id[br.mappings[dest][inds]])['acronym']
    return remapped_acronyms


def get_full_region_name(acronyms, brainregions=None):
    """
    Retrieve the full region names corresponding to a list of acronyms.
    This function takes a list of brain region acronyms and returns their full names
    using the provided `BrainRegions` object. If an acronym is not found, it is returned
    as-is in the output.
    Args:
        acronyms (list of str): A list of brain region acronyms to look up.
        brainregions (BrainRegions, optional): An instance of the `BrainRegions` class 
            that contains mapping information for acronyms and full region names. 
            If not provided, a new instance of `BrainRegions` will be created.
    Returns:
        list of str or str: A list of full region names corresponding to the input acronyms.
        If the input list contains only one acronym, the function returns a single string
        instead of a list.
    Raises:
        IndexError: If an acronym is not found in the `BrainRegions` object, it is handled
        by appending the acronym itself to the output list.
    """
    br = brainregions or BrainRegions()
    full_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regname = br.name[np.argwhere(br.acronym == acronym).flatten()][0]
            full_region_names.append(regname)
        except IndexError:
            full_region_names.append(acronym)
    if len(full_region_names) == 1:
        return full_region_names[0]
    else:
        return full_region_names
    
    
def combine_regions(allen_acronyms, split_peri=True, abbreviate=True, brainregions=None):
    """
    Maps Allen Brain Atlas acronyms to broader brain region categories.
    This function remaps a list of Allen Brain Atlas acronyms to broader brain 
    region categories, either in abbreviated or full form. It also allows for 
    splitting the perirhinal cortex into areas 35 and 36.
    Parameters:
    -----------
    allen_acronyms : list or array-like
        A list or array of Allen Brain Atlas acronyms to be remapped.
    split_peri : bool, optional
        If True, splits the perirhinal cortex into areas 35 and 36. 
        Defaults to True.
    abbreviate : bool, optional
        If True, returns abbreviated region names. If False, returns full 
        region names. Defaults to True.
    brainregions : BrainRegions, optional
        An instance of the BrainRegions class. If None, a new instance is 
        created. Defaults to None.
    Returns:
    --------
    regions : numpy.ndarray
        An array of remapped brain region names corresponding to the input 
        acronyms.
    """

    br = brainregions or BrainRegions()
    acronyms = remap(allen_acronyms)  # remap to Beryl
    regions = np.array(['root'] * len(acronyms), dtype=object)
    if abbreviate:
        if split_peri:
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('ECT'))['acronym'])] = 'PERI 36'
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('PERI'))['acronym'])] = 'PERI 35'
        else:
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('ECT'))['acronym'])] = 'PERI'
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('PERI'))['acronym'])] = 'PERI'
        regions[np.in1d(acronyms, br.descendants(br.acronym2id('ENT'))['acronym'])] = 'ENT'
        regions[np.in1d(acronyms, br.descendants(br.acronym2id('VIS'))['acronym'])] = 'VIS'
        regions[np.in1d(acronyms, br.descendants(br.acronym2id('AUD'))['acronym'])] = 'AUD'
        regions[np.in1d(acronyms, br.descendants(br.acronym2id('TEa'))['acronym'])] = 'TEa'
        regions[acronyms == 'CA1'] = 'CA1'
    else:
        if split_peri:
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('ECT'))['acronym'])] = 'Perirhinal cortex (area 36)'
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('PERI'))['acronym'])] = 'Perirhinal cortex (area 35)'
        else:
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('ECT'))['acronym'])] = 'Perirhinal cortex'
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('PERI'))['acronym'])] = 'Perirhinal cortex'
        regions[np.in1d(acronyms, br.descendants(br.acronym2id('ENT'))['acronym'])] = 'Enthorhinal cortex'
        regions[np.in1d(acronyms, br.descendants(br.acronym2id('VIS'))['acronym'])] = 'Visual cortex'
        regions[np.in1d(acronyms, br.descendants(br.acronym2id('AUD'))['acronym'])] = 'Auditory cortex'
        regions[np.in1d(acronyms, br.descendants(br.acronym2id('TEa'))['acronym'])] = 'Temporal association area'
        regions[acronyms == 'CA1'] = 'CA1'     

    return regions


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


def bin_signal(timestamps, signal, bin_edges):
    """
    Bin a signal based on provided timestamps and bin edges.
    This function groups the `signal` values into bins defined by `bin_edges` 
    based on their corresponding `timestamps`. It computes the mean value of 
    the signal within each bin.
    Parameters:
    -----------
    timestamps : numpy.ndarray
        Array of timestamps corresponding to the signal values.
    signal : numpy.ndarray
        Array of signal values to be binned.
    bin_edges : numpy.ndarray
        Array defining the edges of the bins. The bins are defined as 
        [bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges) - 1).
    Returns:
    --------
    numpy.ndarray
        Array of mean signal values for each bin. If a bin contains no 
        timestamps, its mean value will be zero.
    """

    bin_indices = np.digitize(timestamps[(timestamps >= bin_edges[0]) & (timestamps < bin_edges[-1])],
                              bins=bin_edges, right=False) - 1
    bin_sums = np.bincount(
        bin_indices,
        weights=signal[(timestamps >= bin_edges[0]) & (timestamps < bin_edges[-1])],
        minlength=len(bin_edges) - 1
        )
    bin_means = np.divide(bin_sums, np.bincount(bin_indices), out=np.zeros_like(bin_sums),
                          where=np.bincount(bin_indices)!=0)
    return bin_means


def event_aligned_averages(signal, timestamps, events, timebins, return_df=False):
    """
    Aligns a 1D signal to events and computes the average for each time bin relative to the events.

    Parameters:
    -----------
    signal : numpy.ndarray
        1D array of signal values.
    timestamps : numpy.ndarray
        1D array of timestamps corresponding to the signal values.
    events : numpy.ndarray
        1D array of event timestamps.
    timebins : numpy.ndarray
        1D array of time bin edges (in seconds) relative to the events.
    return_df : boolean
        Wether to return the result as a dataframe

    Returns:
    --------
    averages : numpy.ndarray
        2D array of shape (len(events), len(timebins) - 1) containing the average signal
        for each event and time bin.
        
    OR
    
    df_long : DataFrame
        A long form DataFrame
    """
    num_bins = len(timebins) - 1
    num_events = len(events)
    averages = np.zeros((num_events, num_bins))

    for i, event in enumerate(events):
        # Shift timebins relative to the current event
        bin_edges = timebins + event
        # Mask signal values within the bin edges
        for j in range(num_bins):
            bin_mask = (timestamps >= bin_edges[j]) & (timestamps < bin_edges[j + 1])
            averages[i, j] = np.mean(signal[bin_mask]) if np.any(bin_mask) else 0

    if return_df:
        time_ax = timebins[:-1] + (np.mean(np.diff(timebins)) / 2)
        df_long = pd.DataFrame({
            'event': np.repeat(np.arange(events.shape[0]), time_ax.shape[0]),
            'time': np.tile(time_ax, events.shape[0]),  
            'value': averages.flatten()
        })
        return df_long
    else:
        return averages


def peri_event_trace(array, timestamps, event_times, event_ids, ax, t_before=1, t_after=3,
                     event_labels=None, color_palette='colorblind', ind_lines=False, kwargs={}):

    # Construct dataframe for plotting
    plot_df = pd.DataFrame()
    samp_rate = np.round(np.mean(np.diff(timestamps)), 3)
    time_x = np.arange(0, t_before + t_after, samp_rate)
    time_x = time_x - time_x[int(t_before * (1/samp_rate))]
    for i, t in enumerate(event_times[~np.isnan(event_times)]):
        zero_point = np.argmin(np.abs(timestamps - t))
        this_array = array[zero_point - np.sum(time_x < 0) : (zero_point + np.sum(time_x > 0)) + 1]
        if this_array.shape[0] != time_x.shape[0]:
            print('Trial time mismatch')
            continue
        plot_df = pd.concat((plot_df, pd.DataFrame(data={
            'y': this_array, 'time': time_x, 'event_id': event_ids[i], 'event_nr': i})))

    # Plot
    if ind_lines:
        sns.lineplot(data=plot_df, x='time', y='y', hue='event_id', estimator=None, units='event_nr',
                     palette=color_palette, err_kws={'lw': 0}, ax=ax, **kwargs)
    else:
        sns.lineplot(data=plot_df, x='time', y='y', hue='event_id', errorbar='se',
                     palette=color_palette, err_kws={'lw': 0}, ax=ax, **kwargs)
    if event_labels is None:
        ax.get_legend().remove()
    else:
        g = ax.legend(title='', prop={'size': 5.5})
        for t, l in zip(g.texts, event_labels):
            t.set_text(l)

    # sns.despine(trim=True)
    # plt.tight_layout()


def get_spike_counts_in_bins(spike_times, spike_clusters, intervals):
    """
    From ibllib package 
    
    Return the number of spikes in a sequence of time intervals, for each neuron.

    Parameters
    ----------
    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    intervals : 2D array of shape (n_events, 2)
        the start and end times of the events

    Returns
    ---------
    counts : 2D array of shape (n_neurons, n_events)
        the spike counts of all neurons ffrom scipy.stats import sem, tor all events
        value (i, j) is the number of spikes of neuron `neurons[i]` in interval #j
    cluster_ids : 1D array
        list of cluster ids
    """

    # Check input
    assert intervals.ndim == 2
    assert intervals.shape[1] == 2

    intervals_idx = np.searchsorted(spike_times, intervals)

    # For each neuron and each interval, the number of spikes in the interval.
    cluster_ids = np.unique(spike_clusters)
    n_neurons = len(cluster_ids)
    n_intervals = intervals.shape[0]
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    for j in range(n_intervals):
        i0, i1 = intervals_idx[j, :]
        # Count the number of spikes in the window, for each neuron.
        x = np.bincount(spike_clusters[i0:i1], minlength=cluster_ids.max() + 1)
        counts[:, j] = x[cluster_ids]
    return counts, cluster_ids


def calculate_peths(
        spike_times, spike_clusters, cluster_ids, align_times, pre_time=0.2,
        post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True):
    """
    Calculates peri-event time histograms for a set of clusters.

    :param spike_times: Spike times (in seconds).
    :type spike_times: array-like
    :param spike_clusters: Cluster IDs for each spike.
    :type spike_clusters: array-like
    :param cluster_ids: A unique, sorted list of cluster IDs to analyze.
    :type cluster_ids: array-like
    :param align_times: Times (in seconds) to align the PSTHs to.
    :type align_times: array-like
    :param pre_time: Time (in seconds) before the align time.
    :type pre_time: float
    :param post_time: Time (in seconds) after the align time.
    :type post_time: float
    :param bin_size: Width of time bins (in seconds).
    :type bin_size: float
    :param smoothing: Standard deviation (in seconds) of Gaussian kernel for smoothing.
                      Set to 0 for no smoothing.
    :type smoothing: float
    :param return_fr: If True, returns firing rate (spikes/s). If False, returns spike counts.
    :type return_fr: bool
    :return: A tuple of (peths, binned_spikes).
             peths is a dictionary containing the mean, std, time scale, and cluster scale.
             binned_spikes is a 3D numpy array (n_trials, n_clusters, n_bins).
    """
    # Ensure cluster_ids are unique and sorted, as this is assumed by the binning logic.
    ids = np.unique(cluster_ids)
    
    # Create a mapping from the original cluster ID to an index (0, 1, 2, ...)
    cluster_map = {cid: i for i, cid in enumerate(ids)}

    # --- BUG FIX: Use np.linspace for robust bin creation ---
    # Calculate the number of bins required in a numerically stable way
    n_bins = int(round((pre_time + post_time) / bin_size))
    # Create the bin edges using linspace to avoid floating point inaccuracies of arange
    t_bin_edges = np.linspace(-pre_time, post_time, n_bins + 1)

    # Initialize the main data array
    binned_spikes = np.zeros((len(align_times), len(ids), n_bins))

    # Pre-filter spikes to only those belonging to the clusters of interest
    idx_spikes_in_clusters = np.isin(spike_clusters, ids)
    s_times, s_clusts = spike_times[idx_spikes_in_clusters], spike_clusters[idx_spikes_in_clusters]

    # Map the cluster IDs of the filtered spikes to their new indices (0, 1, 2, ...)
    if len(s_clusts) > 0:
        s_clusts_mapped = np.array([cluster_map[c] for c in s_clusts])
    else:
        s_clusts_mapped = np.array([])


    # Define the bins for the y-axis of the 2D histogram (the clusters)
    y_bins = np.arange(len(ids) + 1) - 0.5

    # Iterate over each alignment time (trial)
    for i, t_0 in enumerate(align_times):
        # Define the absolute time bin edges for the current trial
        x_bins = t_bin_edges + t_0
        
        # Find the indices of spikes that fall ONLY within this trial's time window
        trial_indices = np.where((s_times >= x_bins[0]) & (s_times < x_bins[-1]))[0]

        # Select only the spikes and clusters for this specific trial
        trial_s_times = s_times[trial_indices]
        trial_s_clusts_mapped = s_clusts_mapped[trial_indices]
        
        # Use np.histogram2d to efficiently bin the spikes for this trial.
        # It handles empty spike arrays correctly (returns all zeros).
        counts, _, _ = np.histogram2d(
            trial_s_times,           # Spikes for THIS trial only
            trial_s_clusts_mapped,   # Clusters for THIS trial only
            bins=[x_bins, y_bins]     # Time and cluster bins for this trial
        )
        
        # The result needs to be transposed and stored in the main array
        binned_spikes[i, :, :] = counts.T

    # If smoothing is requested, apply it now to the binned counts
    if smoothing > 0 and smoothing > bin_size: # smoothing must be greater than bin_size
        win_size = int(np.ceil(4 * (smoothing / bin_size)))
        window = gaussian(win_size, std=smoothing / bin_size)
        window /= np.sum(window)

        for i in range(binned_spikes.shape[0]):
            for j in range(binned_spikes.shape[1]):
                binned_spikes[i, j, :] = convolve(binned_spikes[i, j, :], window, mode='same')

    # Convert to firing rate if requested
    if return_fr:
        binned_spikes /= bin_size

    # Calculate final means and stds across trials
    peth_means = np.mean(binned_spikes, axis=0)
    peth_stds = np.std(binned_spikes, axis=0)

    # Final time scale should be the center of each bin
    tscale_final = t_bin_edges[:-1] + (bin_size / 2)

    # Package the output
    peths = {
        'means': peth_means,
        'stds': peth_stds,
        'tscale': tscale_final,
        'cscale': ids
    }

    return peths, binned_spikes


def peri_multiple_events_time_histogram(
        spike_times, spike_clusters, events, event_ids, cluster_id,
        t_before=0.2, t_after=0.5, bin_size=0.025, smoothing=0.025, as_rate=True,
        include_raster=False, error_bars='sem', ax=None, ylim=None,
        pethline_kwargs=[{'color': 'blue', 'lw': 2}, {'color': 'red', 'lw': 2}],
        errbar_kwargs=[{'color': 'blue', 'alpha': 0.5}, {'color': 'red', 'alpha': 0.5}],
        raster_kwargs=[{'color': 'blue', 'lw': 0.5}, {'color': 'red', 'lw': 0.5}],
        eventline_kwargs={'color': 'black', 'alpha': 0.5}, **kwargs):
    """
    From ibllib package

    Plot peri-event time histograms, with the meaning firing rate of units centered on a given
    series of events. Can optionally add a raster underneath the PETH plot of individual spike
    trains about the events.

    Parameters
    ----------
    spike_times : array_like
        Spike times (in seconds)
    spike_clusters : array-like
        Cluster identities for each element of spikes
    events : array-like
        Times to align the histogram(s) to
    event_ids : array-like
        Identities of events
    cluster_id : int
        Identity of the cluster for which to plot a PETH

    t_before : float, optional
        Time before event to plot (default: 0.2s)
    t_after : float, optional
        Time after event to plot (default: 0.5s)
    bin_size :float, optional
        Width of bin for histograms (default: 0.025s)
    smoothing : float, optional
        Sigma of gaussian smoothing to use in histograms. (default: 0.025s)
    as_rate : bool, optional
        Whether to use spike counts or rates in the plot (default: `True`, uses rates)
    include_raster : bool, optional
        Whether to put a raster below the PETH of individual spike trains (default: `False`)
    error_bars : {'std', 'sem', 'none'}, optional
        Defines which type of error bars to plot. Options are:
        -- `'std'` for 1 standard deviation
        -- `'sem'` for standard error of the mean
        -- `'none'` for only plotting the mean value
        (default: `'std'`)
    ax : matplotlib axes, optional
        If passed, the function will plot on the passed axes. Note: current
        behavior causes whatever was on the axes to be cleared before plotting!
        (default: `None`)
    ylim : float, optional
        Set the limit of the y-axis, if None the max of the PSTH will be used
    pethline_kwargs : dict, optional
        Dict containing line properties to define PETH plot line. Default
        is a blue line with weight of 2. Needs to have color. See matplotlib plot documentation
        for more options.
        (default: `{'color': 'blue', 'lw': 2}`)
    errbar_kwargs : dict, optional
        Dict containing fill-between properties to define PETH error bars.
        Default is a blue fill with 50 percent opacity.. Needs to have color. See matplotlib
        fill_between documentation for more options.
        (default: `{'color': 'blue', 'alpha': 0.5}`)
    eventline_kwargs : dict, optional
        Dict containing fill-between properties to define line at event.
        Default is a black line with 50 percent opacity.. Needs to have color. See matplotlib
        vlines documentation for more options.
        (default: `{'color': 'black', 'alpha': 0.5}`)
    raster_kwargs : dict, optional
        Dict containing properties defining lines in the raster plot.
        Default is black lines with line width of 0.5. See matplotlib vlines for more options.
        (default: `{'color': 'black', 'lw': 0.5}`)

    Returns
    -------
        ax : matplotlib axes
            Axes with all of the plots requested.
    """

    # Check to make sure if we fail, we fail in an informative way
    if not len(spike_times) == len(spike_clusters):
        raise ValueError('Spike times and clusters are not of the same shape')
    if len(events) == 1:
        raise ValueError('Cannot make a PETH with only one event.')
    if error_bars not in ('std', 'sem', 'none'):
        raise ValueError('Invalid error bar type was passed.')
    if not all(np.isfinite(events)):
        raise ValueError('There are NaN or inf values in the list of events passed. '
                         ' Please remove non-finite data points and try again.')

    # Construct an axis object if none passed
    if ax is None:
        plt.figure()
        ax = plt.gca()
    # Plot the curves and add error bars
    mean_max, bars_max = [], []
    for i, event_id in enumerate(np.unique(event_ids)):
        # Compute peths
        peths, binned_spikes = singlecell.calculate_peths(spike_times, spike_clusters, [cluster_id],
                                                          events[event_ids == event_id], t_before,
                                                          t_after, bin_size, smoothing, as_rate)
        mean = peths.means[0, :]
        ax.plot(peths.tscale, mean, **pethline_kwargs[i])
        if error_bars == 'std':
            bars = peths.stds[0, :]
        elif error_bars == 'sem':
            bars = peths.stds[0, :] / np.sqrt(np.sum(event_ids == event_id))
        else:
            bars = np.zeros_like(mean)
        if error_bars != 'none':
            ax.fill_between(peths.tscale, mean - bars, mean + bars, **errbar_kwargs[i])
        mean_max.append(mean.max())
        bars_max.append(bars[mean.argmax()])

    # Plot the event marker line. Extends to 5% higher than max value of means plus any error bar.
    if ylim is None:
        plot_edge = (np.max(mean_max) + bars_max[np.argmax(mean_max)]) * 1.05
    else:
        plot_edge = ylim
    ax.vlines(0., 0., plot_edge, **eventline_kwargs)
    # Set the limits on the axes to t_before and t_after. Either set the ylim to the 0 and max
    # values of the PETH, or if we want to plot a spike raster below, create an equal amount of
    # blank space below the zero where the raster will go.
    ax.set_xlim([-t_before, t_after])
    ax.set_ylim([-plot_edge if include_raster else 0., plot_edge])
    # Put y ticks only at min, max, and zero
    if mean.min() != 0:
        ax.set_yticks([0, mean.min(), mean.max()])
    else:
        ax.set_yticks([0., mean.max()])
    # Move the x axis line from the bottom of the plotting space to zero if including a raster,
    # Then plot the raster
    if include_raster:
        ax.axhline(0., color='black', lw=0.5)
        tickheight = plot_edge / len(events)  # How much space per trace
        tickedges = np.arange(0., -plot_edge - 1e-5, -tickheight)
        clu_spks = spike_times[spike_clusters == cluster_id]
        ii = 0
        for k, event_id in enumerate(np.unique(event_ids)):
            for i, t in enumerate(events[event_ids == event_id]):
                idx = np.bitwise_and(clu_spks >= t - t_before, clu_spks <= t + t_after)
                event_spks = clu_spks[idx]
                ax.vlines(event_spks - t, tickedges[i + ii + 1], tickedges[i + ii],
                          **raster_kwargs[k])
            ii += np.sum(event_ids == event_id)
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes', y=0.75)
    else:
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s) after event')
    return ax


def circ_shift(series1, series2, n_shifts=10000, min_shift_percentage=0.05):
    """
    Calculates the Pearson correlation between two time series and assesses its
    significance using a linear (circular) shift permutation test.

    This method preserves the auto-correlation within each series while
    breaking the temporal relationship between them to form a null distribution.

    Parameters:
    ----------
    series1 : np.ndarray
        First time series (e.g., binned spike counts).
    series2 : np.ndarray
        Second time series of the same length (e.g., running speed).
    n_shifts : int, optional
        Number of random shifts to perform for the null distribution.
        Default is 10000.
    min_shift_percentage : float, optional
        The minimum percentage of the series length to shift by. This helps
        avoid very small shifts that might not adequately break the relationship
        if there's very slow autocorrelation. Value between 0 and 0.5.
        Default is 0.05 (5%). If set to 0, shifts can be as small as 1.
    verbose : bool, optional
        Whether to print results. Default is True.

    Returns:
    -------
    observed_r : float
        The Pearson correlation coefficient of the original data.
    p_value : float
        The p-value calculated from the shifted null distribution.
    null_distribution_r : np.ndarray
        An array of correlation coefficients from the shifted data.
    """
    if len(series1) != len(series2):
        raise ValueError("Input series must have the same length.")
    if len(series1) < 2:
        raise ValueError("Input series must have at least 2 data points.")

    n = len(series1)

    # 1. Calculate the observed correlation
    observed_r, _ = pearsonr(series1, series2) # We don't need the parametric p-value from pearsonr

    # 2. Generate the null distribution by circularly shifting one series
    null_distribution_r = np.zeros(n_shifts)
    
    # Determine the minimum and maximum shift amounts
    min_shift = max(1, int(n * min_shift_percentage))
    max_shift = n - min_shift # Symmetrical to avoid wrapping back to small effective shifts

    if min_shift >= max_shift : # Handles very short series or high min_shift_percentage
        print(f"Warning: min_shift ({min_shift}) is too close to series length ({n}). "
              f"Using shifts between 1 and n-1.")
        possible_shifts = np.arange(1, n)
    else:
        possible_shifts = np.concatenate((
            np.arange(min_shift, max_shift + 1),
        ))
        if len(possible_shifts) == 0: # Fallback for extremely short series
             possible_shifts = np.arange(1, n)

    for i in range(n_shifts):
        # Randomly choose a shift amount (excluding 0, and respecting min_shift)
        shift_amount = np.random.choice(possible_shifts)
        
        # Circularly shift series1
        shifted_series1 = np.roll(series1, shift_amount)
        
        # Calculate correlation for this shifted pair
        r_null, _ = pearsonr(shifted_series1, series2)
        null_distribution_r[i] = r_null
        
    # The p-value is the sum of probabilities in both tails if looking for any modulation
    p_value = np.sum(np.abs(null_distribution_r) >= np.abs(observed_r)) / n_shifts

    return observed_r, p_value, null_distribution_r


def load_lfp(probe_path, channels, timewin='passive'):
    """
    """
    
    # Import SpikeInterface
    import spikeinterface.full as si
    
    # Load in raw data using SpikeInterface 
    rec = si.read_cbin_ibl(probe_path)
    
    # Filter out LFP band
    rec_lfp = si.bandpass_filter(rec, freq_min=1, freq_max=400)
    
    # Correct for inter-sample shift
    rec_shifted = si.phase_shift(rec_lfp)    
    
    # Interpolate over bad channels  
    rec_car_temp = si.common_reference(rec_lfp)
    _, all_channels = si.detect_bad_channels(
        rec_car_temp, method='mad', std_mad_threshold=3, seed=42)
    noisy_channel_ids = rec_car_temp.get_channel_ids()[all_channels == 'noise']           
    rec_interpolated = si.interpolate_bad_channels(rec_shifted, noisy_channel_ids)
    
    # Do common average reference
    rec_car = si.common_reference(rec_interpolated)
    
    # Downsample to 2500 Hz
    rec_final = si.resample(rec_car, 2500)
        
    # Load in trials
    trials = pd.read_csv(join(probe_path, '..', '..', 'trials.csv'))
    
    # Get LFP from the requested channels for the passive period
    time_start = 3300
    time_end = 3300 + 120
    samples_start = int(time_start * rec_final.sampling_frequency)
    samples_end = int(time_end * rec_final.sampling_frequency)
    lfp_traces = rec_final.get_traces(start_frame=samples_start, end_frame=samples_end, 
                                      channel_ids=channels)
    
    
    # Load in LFP
    sr = spikeglx.Reader(join(probe_path, '_spikeGLX_ephysData_g0_t0.imec1.lf.bin'))
    #time_start = trials.loc[trials.index[-1], 'exitEnvTime'] + 60
    #time_end = sr.shape[0] / sr.fs
    time_start = 3300
    time_end = 3300 + 120
    samples_start = int(time_start * sr.fs)
    samples_end = int(time_end * sr.fs)
    signal = sr.read(nsel=slice(samples_start, samples_end, None), csel=channels)[0]
    time = np.arange(samples_start, samples_end) / sr.fs
    
    # Do common average reference
    common_avg = np.median(signal, axis=1, keepdims=True)
    signal_car = signal - common_avg
    
    try:
        lfp_paths, _ = one.load_datasets(eid, download_only=True, datasets=[
            '_spikeglx_ephysData_g*_t0.imec*.lf.cbin', '_spikeglx_ephysData_g*_t0.imec*.lf.meta',
            '_spikeglx_ephysData_g*_t0.imec*.lf.ch'], collections=[f'raw_ephys_data/{probe}'] * 3)
        lfp_path = lfp_paths[0]
        sr = spikeglx.Reader(lfp_path)
    except Exception:
        return [], []
    
    # Load in trials    
    try:
        trials = one.load_object(eid, 'trials')
        trial_times = trials['stimOn_times'][~np.isnan(trials['stimOn_times'])]
    except Exception:
        print('Could not load trial table, trying another way')
        if 'alf/_ibl_trials.stimOn_times.npy' in one.list_datasets(eid):
            trial_times = one.load_dataset(eid, '_ibl_trials.stimOn_times.npy')
        elif 'alf/_ibl_trials.stimOff_times.npy' in one.list_datasets(eid):
            trial_times = one.load_dataset(eid, '_ibl_trials.stimOff_times.npy')[:-1]
        elif 'alf/_ibl_trials.goCue_times.npy' in one.list_datasets(eid):
            trial_times = one.load_dataset(eid, '_ibl_trials.goCue_times.npy')
        else:
            print('No luck, just using the last 15 minutes of the recording')
            trial_times = (sr.shape[0] / sr.fs) - (60 * 15)
    trial_times = trial_times[~np.isnan(trial_times)]
    
    if timewin == 'spont':
        # Take 10 minutes of spontaneous activity starting 5 minutes after the last trial
        time_start = trial_times[-1] + (60 * 5)
        time_end = trial_times[-1] + (60 * 15)
    elif timewin == 'passive':
        # Take all time after the last trial until the end of the recording
        time_start = trial_times[-1] + 60
        time_end = sr.shape[0] / sr.fs
    elif timewin == 'task':
        # Take all time during the task
        time_start = 0
        time_end = trial_times[-1]
    
    # Convert seconds to samples
    samples_start = int(time_start * sr.fs)
    samples_end = int(time_end * sr.fs)
    
    # Load in lfp slice
    signal = sr.read(nsel=slice(samples_start, samples_end, None), csel=channels)[0]
    time = np.arange(samples_start, samples_end) / sr.fs

    if signal.shape[0] == 0:
        return [], []

    return signal, time