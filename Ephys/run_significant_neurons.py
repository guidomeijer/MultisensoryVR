# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:45:16 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from joblib import Parallel, delayed
from zetapy import zetatest, zetatest2
from msvr_functions import paths, load_neural_data, load_subjects, load_objects

# Settings
T_BEFORE = 1  # s
T_AFTER = 2
ALPHA = 0.05
OVERWRITE = True
max_dur = T_BEFORE + T_AFTER

# Initialize
path_dict = paths()
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# Load in previous data
if OVERWRITE:
    stats_df = pd.DataFrame()
else:
    stats_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
    stats_df[['subject', 'date', 'probe']] = stats_df[['subject', 'date', 'probe']].astype(str)
    merged = rec.merge(stats_df, on=['subject', 'date', 'probe'], how='left', indicator=True)
    rec = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    

# %% Function for parallel processing

def run_zetatest2(neuron_id, event_times1, event_times2):
    these_spikes = spikes['times'][spikes['clusters'] == neuron_id]
    try:
        p_value, zeta_dict = zetatest2(these_spikes, event_times1, these_spikes, event_times2,
                                       dblUseMaxDur=max_dur)
        zeta_score = zeta_dict['dblZETA']
    except Exception:
        p_value = np.nan    
        zeta_score = np.nan        
    return p_value, zeta_score

def run_zetatest(neuron_id, event_times):
    these_spikes = spikes['times'][spikes['clusters'] == neuron_id]
    p_value, zeta_dict, _ = zetatest(these_spikes, event_times, dblUseMaxDur=max_dur)
    zeta_score = zeta_dict['dblZETA']
    return p_value, zeta_score

# %%
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\nStarting {subject} {date} {probe}..')
    
    # Load in data
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe, histology=True, only_good=True,
                                                  min_fr=0.1)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)
   
    # The ZETA test starts looking for significant differences from timepoint zero so we need to 
    # subtract the pre time from all the onset times
    all_obj_df['times'] = all_obj_df['times'] - T_BEFORE
    
    # Get event times 
    goal1_times = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 1), 'times']
    no_goal1_times = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 0), 'times']
    goal2_times = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 1), 'times']
    no_goal2_times = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 0), 'times']
    sound1_times = all_obj_df.loc[(all_obj_df['object'] == 3) & (all_obj_df['sound'] == 1), 'times']
    sound2_times = all_obj_df.loc[(all_obj_df['object'] == 3) & (all_obj_df['sound'] == 2), 'times']
    
    # Difference between contexts for the second rewarded object (goal)
    print('Calculating difference between goal/distractor context')
    results = Parallel(n_jobs=-1)(
        delayed(run_zetatest2)(neuron_id, goal1_times, no_goal1_times)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    goal1_p = np.array([result[0] for result in results])
    goal1_z = np.array([result[1] for result in results])
    results = Parallel(n_jobs=-1)(
        delayed(run_zetatest2)(neuron_id, goal2_times, no_goal2_times)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    goal2_p = np.array([result[0] for result in results])
    goal2_z = np.array([result[1] for result in results])
    
    # Difference between the two sounds for the control object
    print('Calculating difference between sound context for control object')
    results = Parallel(n_jobs=-1)(
        delayed(run_zetatest2)(neuron_id, sound1_times, sound2_times)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    control_sound_p = np.array([result[0] for result in results])
    control_sound_z = np.array([result[1] for result in results])
    
    # Get neurons which significantly respond to any object appearance 
    print('Calculating significant object onset responses')
    results = Parallel(n_jobs=-1)(
        delayed(run_zetatest)(neuron_id, all_obj_df['times'])
        for i, neuron_id in enumerate(clusters['cluster_id']))
    obj_p = np.array([result[0] for result in results])
    obj_z = np.array([result[1] for result in results])
    
    # Difference between the pairs of two objects (per sound)
    print('Calculating difference between objects')
    results = Parallel(n_jobs=-1)(
        delayed(run_zetatest2)(
            neuron_id,
            all_obj_df.loc[(all_obj_df['sound'] == 1) & (all_obj_df['object'] == 1), 'times'],
            all_obj_df.loc[(all_obj_df['sound'] == 1) & (all_obj_df['object'] == 2), 'times'])
        for i, neuron_id in enumerate(clusters['cluster_id']))
    obj12_sound1_p = np.array([result[0] for result in results])
    
    results = Parallel(n_jobs=-1)(
        delayed(run_zetatest2)(
            neuron_id,
            all_obj_df.loc[(all_obj_df['sound'] == 1) & (all_obj_df['object'] == 1), 'times'],
            all_obj_df.loc[(all_obj_df['sound'] == 1) & (all_obj_df['object'] == 3), 'times'])
        for i, neuron_id in enumerate(clusters['cluster_id']))
    obj13_sound1_p = np.array([result[0] for result in results])
    
    results = Parallel(n_jobs=-1)(
        delayed(run_zetatest2)(
            neuron_id,
            all_obj_df.loc[(all_obj_df['sound'] == 1) & (all_obj_df['object'] == 2), 'times'],
            all_obj_df.loc[(all_obj_df['sound'] == 1) & (all_obj_df['object'] == 3), 'times'])
        for i, neuron_id in enumerate(clusters['cluster_id']))
    obj23_sound1_p = np.array([result[0] for result in results])
    
    results = Parallel(n_jobs=-1)(
        delayed(run_zetatest2)(
            neuron_id,
            all_obj_df.loc[(all_obj_df['sound'] == 2) & (all_obj_df['object'] == 1), 'times'],
            all_obj_df.loc[(all_obj_df['sound'] == 2) & (all_obj_df['object'] == 2), 'times'])
        for i, neuron_id in enumerate(clusters['cluster_id']))
    obj12_sound2_p = np.array([result[0] for result in results])
    
    results = Parallel(n_jobs=-1)(
        delayed(run_zetatest2)(
            neuron_id,
            all_obj_df.loc[(all_obj_df['sound'] == 2) & (all_obj_df['object'] == 1), 'times'],
            all_obj_df.loc[(all_obj_df['sound'] == 2) & (all_obj_df['object'] == 3), 'times'])
        for i, neuron_id in enumerate(clusters['cluster_id']))
    obj13_sound2_p = np.array([result[0] for result in results])
    
    results = Parallel(n_jobs=-1)(
        delayed(run_zetatest2)(
            neuron_id,
            all_obj_df.loc[(all_obj_df['sound'] == 2) & (all_obj_df['object'] == 2), 'times'],
            all_obj_df.loc[(all_obj_df['sound'] == 2) & (all_obj_df['object'] == 3), 'times'])
        for i, neuron_id in enumerate(clusters['cluster_id']))
    obj23_sound2_p = np.array([result[0] for result in results])
    
    # Get significance
    sig_goal = (goal1_p < ALPHA) | (goal2_p < ALPHA)
    sig_control = control_sound_p < ALPHA
    sig_obj_onset = obj_p < ALPHA
    sig_obj_diff = ((obj12_sound1_p < ALPHA / 6) | (obj13_sound1_p < ALPHA / 6) | (obj23_sound1_p < ALPHA / 6)
                    | (obj12_sound1_p < ALPHA / 6) | (obj13_sound1_p < ALPHA / 6) | (obj23_sound1_p < ALPHA / 6))
    
    # Add to dataframe
    stats_df = pd.concat((stats_df, pd.DataFrame(data={
        'subject': subject, 'date': date, 'probe': probe, 'neuron_id': clusters['cluster_id'],
        'region': clusters['region'], 'allen_acronym': clusters['acronym'],
        'x': clusters['x'], 'y': clusters['y'], 'z': clusters['z'],
        'sig_goal': sig_goal, 'sig_obj_onset': sig_obj_onset, 'sig_control': sig_control,
        'sig_obj_diff': sig_obj_diff, 'p_goal': np.min(np.vstack((goal1_p, goal2_p)), axis=0),
        'z_goal': np.max(np.vstack((goal1_z, goal2_z)), axis=0), 
        'p_control_sound': control_sound_p, 'p_obj_onset': obj_p, 'z_obj_onset': obj_z,
        'p_obj12_sound1': obj12_sound1_p, 'p_obj23_sound1': obj23_sound1_p, 'p_obj13_sound1': obj13_sound1_p,
        'p_obj12_sound2': obj12_sound2_p, 'p_obj23_sound2': obj23_sound2_p, 'p_obj13_sound2': obj13_sound2_p
        })))
        
    # Save to disk
    stats_df.to_csv(join(path_dict['save_path'], 'significant_neurons.csv'), index=False)
        
           
        
    
    
