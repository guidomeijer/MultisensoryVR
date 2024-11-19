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
import json, shutil, datetime
from glob import glob
from os.path import join, realpath, dirname, isfile, split, isdir
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
            
            # Copy data from server to local folder
            subjects = os.listdir(join(path_dict['server_path'], 'Subjects'))
            for i, subject in enumerate(subjects):
                if not isdir(join(path_dict['local_data_path'], 'Subjects', subject)):
                    os.mkdir(join(path_dict['local_data_path'], 'Subjects', subject))
                sessions = os.listdir(join(path_dict['server_path'], 'Subjects', subject))
                for j, session in enumerate(sessions):
                    files = [f for f in os.listdir(join(path_dict['server_path'], 'Subjects', subject, session))
                             if (isfile(join(path_dict['server_path'], 'Subjects', subject, session, f))
                                 & (f[-4:] != 'flag'))]
                    if len(files) == 0:
                        continue
                    if not isdir(join(path_dict['local_data_path'], 'Subjects', subject, session)):
                        os.mkdir(join(path_dict['local_data_path'], 'Subjects', subject, session))
                    if not isfile(join(path_dict['local_data_path'], 'Subjects', subject, session, files[0])):
                        print(
                            f'Copying files {join(path_dict["server_path"], "Subjects", subject, session)}')
                        for f, file in enumerate(files):
                            if not isfile(join(path_dict['local_data_path'], 'Subjects', subject, session, file)):
                                shutil.copyfile(join(path_dict['server_path'], 'Subjects', subject, session, file),
                                                join(path_dict['local_data_path'], 'Subjects', subject, session, file))
                        if ((isdir(join(path_dict['server_path'], 'Subjects', subject, session, 'probe00')))
                                & (~isdir(join(path_dict['local_data_path'], 'Subjects', subject, session, 'probe00')))):
                            shutil.copytree(join(path_dict['server_path'], 'Subjects', subject, session, 'probe00'),
                                            join(path_dict['local_data_path'], 'Subjects', subject, session, 'probe00'))
                        if ((isdir(join(path_dict['server_path'], 'Subjects', subject, session, 'probe01')))
                                & (~isdir(join(path_dict['local_data_path'], 'Subjects', subject, session, 'probe01')))):
                            shutil.copytree(join(path_dict['server_path'], 'Subjects', subject, session, 'probe01'),
                                            join(path_dict['local_data_path'], 'Subjects', subject, session, 'probe01'))
                    if (not isdir(join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_video_data'))) & full_sync:
                        print(
                            f'Copying raw video data {join(path_dict["server_path"], "Subjects", subject, session)}')
                        shutil.copytree(join(path_dict['server_path'], 'Subjects', subject, session, 'raw_video_data'),
                                        join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_video_data'))
                    if isdir(join(path_dict['server_path'], 'Subjects', subject, session, 'raw_ephys_data')) & full_sync:
                        if not isdir(join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_ephys_data')):
                            print(
                                f'Copying raw ephys data {join(path_dict["server_path"], "Subjects", subject, session)}')
                            shutil.copytree(join(path_dict['server_path'], 'Subjects', subject, session, 'raw_ephys_data'),
                                            join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_ephys_data'))
                    if (not isdir(join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_behavior_data'))) & full_sync:
                        print(
                            f'Copying raw behavior data {join(path_dict["server_path"], "Subjects", subject, session)}')
                        shutil.copytree(join(path_dict['server_path'], 'Subjects', subject, session, 'raw_behavior_data'),
                                        join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_behavior_data'))

            # Update synchronization timestamp
            with open(join(path_dict['local_data_path'], 'sync_timestamp.txt'), 'w') as f:
                f.write(datetime.datetime.now().strftime('%Y%m%d_%H%M'))
                f.close()
            print('Done')

    return path_dict


def load_objects(subject, date, reorder=True):
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
    trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', subject, date, 'trials.csv'))
    
    # Get reward contingencies
    sound1_obj = subjects.loc[subjects['SubjectID'] == subject, 'Sound1Obj'].values[0]
    sound2_obj = subjects.loc[subjects['SubjectID'] == subject, 'Sound2Obj'].values[0]
    control_obj = subjects.loc[subjects['SubjectID'] == subject, 'ControlObject'].values[0]

    obj1_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 1)[0][0] + 1
    obj2_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 2)[0][0] + 1
    obj3_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 3)[0][0] + 1

    if (trials['positionObj1'][0] > trials['positionObj2'][0]) & reorder:
        # Object 1 comes after object 2
        sound1_obj_ordered = 2
        sound2_obj_ordered = 1
    else:        
        sound1_obj_ordered = 1
        sound2_obj_ordered = 2

    # Prepare trial data
    rew_obj1_df = pd.DataFrame(data={'times': trials[f'enterObj{sound1_obj}'],
                                     'object': sound1_obj_ordered, 'sound': trials['soundId'],
                                     'goal_sound': obj1_goal_sound,
                                     'goal': (trials['soundId'] == obj1_goal_sound).astype(int)})
    rew_obj2_df = pd.DataFrame(data={'times': trials[f'enterObj{sound2_obj}'],
                                     'object': sound2_obj_ordered, 'sound': trials['soundId'],
                                     'goal_sound': obj2_goal_sound,
                                     'goal': (trials['soundId'] == obj2_goal_sound).astype(int)})
    control_obj_df = pd.DataFrame(data={'times': trials[f'enterObj{control_obj}'],
                                        'object': 3, 'sound': trials['soundId'],
                                        'goal_sound': 0,
                                        'goal': (trials['soundId'] == obj3_goal_sound).astype(int)})
    all_obj_df = pd.concat((rew_obj1_df, rew_obj2_df, control_obj_df))
    
    return all_obj_df


def figure_style(font_size=7):
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": font_size,
                "figure.titlesize": font_size,
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
        'HPC': sns.color_palette('Set3')[4],
        'AUD': sns.color_palette('Set3')[4],
        'ENT': sns.color_palette('Set3')[6],
        'CA1': sns.color_palette('Dark2')[4],
        'DG': sns.color_palette('Dark2')[7],
        'TBA': sns.color_palette('tab10')[9],
        'TBA': sns.color_palette('Dark2')[0],
        'TBA': sns.color_palette('Accent')[0],
        'TBA': sns.color_palette('Accent')[1],
        'TBA': sns.color_palette('Accent')[2],
        'TBA': sns.color_palette('tab10')[8],
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
        Whether to only load in neurons that have been manually labelled in Phy.
        The default is True.
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
        
    # Load in cluster data
    clusters = dict()
    clusters['channels'] = np.load(join(session_path, probe, 'clusters.channels.npy'))
    clusters['depths'] = np.load(join(session_path, probe, 'clusters.depths.npy'))
    clusters['amps'] = np.load(join(session_path, probe, 'clusters.amps.npy'))
    clusters['cluster_id'] = np.arange(clusters['channels'].shape[0])
    
    # Add cluster qc metrics
    if isfile(join(session_path, probe, 'clusters.bcUnitType.npy')):
        clusters['bc_label'] = np.load(join(session_path, probe, 'clusters.bcUnitType.npy'),
                                       allow_pickle=True)
    clusters['ks_label'] = pd.read_csv(join(session_path, probe, 'cluster_KSLabel.tsv'),
                                       sep='\t')['KSLabel']
    if isfile(join(session_path, probe, 'cluster_IBLLabel.tsv')):
        clusters['ibl_label'] = pd.read_csv(join(session_path, probe, 'cluster_IBLLabel.tsv'),
                                            sep='\t')['ibl_label']
    if isfile(join(session_path, probe, 'cluster_group.tsv')):
        clusters['manual_label'] = pd.read_csv(join(session_path, probe, 'cluster_group.tsv'),
                                               sep='\t')['group']
        
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
        
        # Use the channel location to infer the brain regions of the clusters
        clusters['acronym'] = channels['acronym'][clusters['channels']]
        clusters['region'] = combine_regions(clusters['acronym'], abbreviate=True, split_peri=True,
                                             brainregions=BrainRegions())
        clusters['full_region'] = combine_regions(clusters['acronym'], abbreviate=False, split_peri=True,
                                                  brainregions=BrainRegions())
        
    # Exclude neurons that are not labelled good or with firing rates which are too low
    select_units = np.ones(clusters['cluster_id'].shape[0]).astype(bool)
    select_units[np.where(clusters['firing_rate'] < min_fr)[0]] = False
    if only_good:     
        if 'manual_label' not in clusters.keys():
            raise Exception('No manual cluster labels found! Set only_good to False to load all neurons.')
        select_units[np.where(clusters['manual_label'] != 'good')[0]] = False
    keep_units = clusters['cluster_id'][select_units]
    spikes['times'] = spikes['times'][np.isin(spikes['clusters'], keep_units)]
    spikes['amps'] = spikes['amps'][np.isin(spikes['clusters'], keep_units)]
    spikes['depths'] = spikes['depths'][np.isin(spikes['clusters'], keep_units)]
    if 'distances' in spikes.keys():
        spikes['distances'] = spikes['distances'][np.isin(spikes['clusters'], keep_units)]
    spikes['clusters'] = spikes['clusters'][np.isin(spikes['clusters'], keep_units)]
    if histology:
        clusters['acronym'] = clusters['acronym'][keep_units]
        clusters['region'] = clusters['region'][keep_units]
        clusters['full_region'] = clusters['full_region'][keep_units]
    clusters['depths'] = clusters['depths'][keep_units]
    clusters['amps'] = clusters['amps'][keep_units]
    clusters['firing_rate'] = clusters['firing_rate'][keep_units]
    clusters['cluster_id'] = clusters['cluster_id'][keep_units]
    
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
    br = brainregions or BrainRegions()
    _, inds = ismember(br.acronym2id(acronyms), br.id[br.mappings[source]])
    remapped_acronyms = br.get(br.id[br.mappings[dest][inds]])['acronym']
    return remapped_acronyms


def get_full_region_name(acronyms, brainregions=None):
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
    
    
def combine_regions(allen_acronyms, split_peri=False, split_hpc=False, abbreviate=True,
                    brainregions=None):
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
        if split_hpc:    
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('DG'))['acronym'])] = 'DG'
            regions[acronyms == 'CA1'] = 'CA1'
        else:
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('DG'))['acronym'])] = 'HPC'
            regions[acronyms == 'CA1'] = 'HPC'
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
        if split_hpc:
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('DG'))['acronym'])] = 'Dentate gyrus'
            regions[acronyms == 'CA1'] = 'CA1'
        else:
            regions[np.in1d(acronyms, br.descendants(br.acronym2id('DG'))['acronym'])] = 'Hippocampus'
            regions[acronyms == 'CA1'] = 'Hippocampus'

    return regions


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


def bin_signal(timestamps, signal, bin_edges):
    bin_indices = np.digitize(timestamps[(timestamps >= bin_edges[0]) & (timestamps <= bin_edges[-1])],
                              bins=bin_edges, right=False) - 1
    bin_sums = np.bincount(
        bin_indices, weights=signal[(timestamps >= bin_edges[0]) & (timestamps <= bin_edges[-1])])
    bin_means = np.divide(bin_sums, np.bincount(bin_indices), out=np.zeros_like(bin_sums),
                          where=np.bincount(bin_indices)!=0)
    return bin_means


def peri_event_trace(array, timestamps, event_times, event_ids, ax, t_before=1, t_after=3,
                     event_labels=None, color_palette='colorblind', ind_lines=False, kwargs=[]):

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
    From ibllib package    

    Calcluate peri-event time histograms; return means and standard deviations
    for each time point across specified clusters

    :param spike_times: spike times (in seconds)
    :type spike_times: array-like
    :param spike_clusters: cluster ids corresponding to each event in `spikes`
    :type spike_clusters: array-like
    :param cluster_ids: subset of cluster ids for calculating peths
    :type cluster_ids: array-like
    :param align_times: times (in seconds) to align peths to
    :type align_times: array-like
    :param pre_time: time (in seconds) to precede align times in peth
    :type pre_time: float
    :param post_time: time (in seconds) to follow align times in peth
    :type post_time: float
    :param bin_size: width of time windows (in seconds) to bin spikes
    :type bin_size: float
    :param smoothing: standard deviation (in seconds) of Gaussian kernel for
        smoothing peths; use `smoothing=0` to skip smoothing
    :type smoothing: float
    :param return_fr: `True` to return (estimated) firing rate, `False` to return spike counts
    :type return_fr: bool
    :return: peths, binned_spikes
    :rtype: peths: Bunch({'mean': peth_means, 'std': peth_stds, 'tscale': ts, 'cscale': ids})
    :rtype: binned_spikes: np.array (n_align_times, n_clusters, n_bins)
    """

    # initialize containers
    n_offset = 5 * int(np.ceil(smoothing / bin_size))  # get rid of boundary effects for smoothing
    n_bins_pre = int(np.ceil(pre_time / bin_size)) + n_offset
    n_bins_post = int(np.ceil(post_time / bin_size)) + n_offset
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), n_bins))

    # build gaussian kernel if requested
    if smoothing > 0:
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)
        # half (causal) gaussian filter
        # window[int(np.ceil(w/2)):] = 0
        window /= np.sum(window)
        binned_spikes_conv = np.copy(binned_spikes)

    ids = np.unique(cluster_ids)

    # filter spikes outside of the loop
    idxs = np.bitwise_and(spike_times >= np.min(align_times) - (n_bins_pre + 1) * bin_size,
                          spike_times <= np.max(align_times) + (n_bins_post + 1) * bin_size)
    idxs = np.bitwise_and(idxs, np.isin(spike_clusters, cluster_ids))
    spike_times = spike_times[idxs]
    spike_clusters = spike_clusters[idxs]

    # compute floating tscale
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    # bin spikes
    for i, t_0 in enumerate(align_times):
        # define bin edges
        ts = tscale + t_0
        # filter spikes
        idxs = np.bitwise_and(spike_times >= ts[0], spike_times <= ts[-1])
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]

        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        xscale = ts
        xind = (np.floor((i_spikes - np.min(ts)) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)

        # store (ts represent bin edges, so there are one fewer bins)
        bs_idxs = np.isin(ids, yscale)
        binned_spikes[i, bs_idxs, :] = r[:, :-1]

        # smooth
        if smoothing > 0:
            idxs = np.where(bs_idxs)[0]
            for j in range(r.shape[0]):
                binned_spikes_conv[i, idxs[j], :] = convolve(
                    r[j, :], window, mode='same', method='auto')[:-1]

    # average
    if smoothing > 0:
        binned_spikes = np.copy(binned_spikes_conv)
    else:
        binned_spikes = np.copy(binned_spikes)
    if return_fr:
        binned_spikes /= bin_size

    peth_means = np.mean(binned_spikes, axis=0)
    peth_stds = np.std(binned_spikes, axis=0)

    if smoothing > 0:
        peth_means = peth_means[:, n_offset:-n_offset]
        peth_stds = peth_stds[:, n_offset:-n_offset]
        binned_spikes = binned_spikes[:, :, n_offset:-n_offset]
        tscale = tscale[n_offset:-n_offset]

    # package output
    tscale = (tscale[:-1] + tscale[1:]) / 2
    peths = dict({'means': peth_means, 'stds': peth_stds, 'tscale': tscale, 'cscale': ids})
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
