# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:45:16 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from zetapy import zetatest, zetatest2
from msvr_functions import paths, load_neural_data, load_subjects, load_objects

# Settings
ALPHA = 0.05
OVERWRITE = True
N_CORES = 16

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


def run_zetatest2(neuron_id, event_times1, event_times2, t_before, t_after, do_shuffle=False):
    these_spikes = spikes['times'][spikes['clusters'] == neuron_id]
    if do_shuffle:
        all_shuffled_events = shuffle(np.concatenate((event_times1, event_times2)))
        event_times1 = all_shuffled_events[:event_times1.shape[0]]
        event_times2 = all_shuffled_events[event_times2.shape[0]:]
    try:
        p_value, zeta_dict = zetatest2(these_spikes, event_times1 - t_before, these_spikes,
                                       event_times2 - t_before, dblUseMaxDur=(t_before + t_after))
        zeta_score = zeta_dict['dblZETA']
    except Exception:
        p_value = np.nan    
        zeta_score = np.nan        
    return p_value, zeta_score


def run_zetatest(neuron_id, event_times, t_before, t_after, do_jitter=False):
    these_spikes = spikes['times'][spikes['clusters'] == neuron_id]
    if do_jitter:
        event_times = np.random.uniform(low=np.min(event_times) - 1, high=np.max(event_times) + 1,
                                        size=event_times.shape[0])
    p_value, zeta_dict, _ = zetatest(these_spikes, event_times - t_before, dblUseMaxDur=(t_before + t_after),
                                     boolParallel=False)
    zeta_score = zeta_dict['dblZETA']
    return p_value, zeta_score


# %%
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\nStarting {subject} {date} {probe} [{i} of {rec.shape[0]}]\n')
    
    # Load in data
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe, histology=True, only_good=True,
                                                  min_fr=0.1)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)      
     
    # Get event times 
    goal1_times = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 1), 'times'].values
    no_goal1_times = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 0), 'times'].values
    goal2_times = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 1), 'times'].values
    no_goal2_times = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 0), 'times'].values
    reward_times = all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['rewarded'] == 1), 'times'].values
    no_reward_times = all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['rewarded'] == 0), 'times'].values
    sound_trials = trials[np.abs(trials['soundOnsetPos'] - trials['enterEnvPos']) < 50]
    sound1_onsets = trials.loc[trials['soundId'] == 1, 'soundOnsetTime'].values
    sound2_onsets = trials.loc[trials['soundId'] == 2, 'soundOnsetTime'].values
        
    # Difference between contexts for the second rewarded object (goal)
    print('Calculating difference between context for object 1')
    results = Parallel(n_jobs=N_CORES)(
        delayed(run_zetatest2)(neuron_id, goal1_times, no_goal1_times, t_before=2, t_after=0)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    goal1_p = np.array([result[0] for result in results])

    results = Parallel(n_jobs=N_CORES)(
        delayed(run_zetatest2)(neuron_id, goal1_times, no_goal1_times, t_before=2, t_after=0, do_shuffle=True)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    goal1_p_shuf = np.array([result[0] for result in results])
            
    print('Calculating difference between context for object 2')
    results = Parallel(n_jobs=N_CORES)(
        delayed(run_zetatest2)(neuron_id, goal2_times, no_goal2_times, t_before=2, t_after=0)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    goal2_p = np.array([result[0] for result in results])

    results = Parallel(n_jobs=N_CORES)(
        delayed(run_zetatest2)(neuron_id, goal2_times, no_goal2_times, t_before=2, t_after=0, do_shuffle=True)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    goal2_p_shuf = np.array([result[0] for result in results])
        
    print('Calculating difference between reward vs no reward')
    results = Parallel(n_jobs=N_CORES)(
        delayed(run_zetatest2)(neuron_id, reward_times, no_reward_times, t_before=0, t_after=2)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    reward_p = np.array([result[0] for result in results])

    results = Parallel(n_jobs=N_CORES)(
        delayed(run_zetatest2)(neuron_id, reward_times, no_reward_times, t_before=0, t_after=2, do_shuffle=True)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    reward_p_shuf = np.array([result[0] for result in results])
    
    # Get neurons which significantly respond to any object appearance 
    print('Calculating significant object responses')
    results = Parallel(n_jobs=N_CORES)(
        delayed(run_zetatest)(neuron_id, all_obj_df['times'], t_before=2, t_after=0)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    obj_p = np.array([result[0] for result in results])
    

    results = Parallel(n_jobs=N_CORES)(
        delayed(run_zetatest)(neuron_id, all_obj_df['times'], t_before=2, t_after=0, do_jitter=True)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    obj_p_shuf = np.array([result[0] for result in results])

    print('Calculating significant difference between sounds at onset')
    results = Parallel(n_jobs=N_CORES)(
        delayed(run_zetatest2)(neuron_id, sound1_onsets, sound2_onsets, t_before=0, t_after=2)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    sound_onset_p = np.array([result[0] for result in results])
     
    results = Parallel(n_jobs=N_CORES)(
        delayed(run_zetatest2)(neuron_id, sound1_onsets, sound2_onsets, t_before=0, t_after=2, do_shuffle=True)
        for i, neuron_id in enumerate(clusters['cluster_id']))
    sound_onset_p_shuf = np.array([result[0] for result in results])
        
    # Add to dataframe
    stats_df = pd.concat((stats_df, pd.DataFrame(data={
        'subject': subject, 'date': date, 'probe': probe, 'neuron_id': clusters['cluster_id'],
        'region': clusters['region'], 'allen_acronym': clusters['acronym'],
        'x': clusters['x'], 'y': clusters['y'], 'z': clusters['z'],
        'p_context_obj1': goal1_p, 'p_context_obj1_shuf': goal1_p_shuf,
        'p_context_obj2': goal2_p, 'p_context_obj2_shuf': goal2_p_shuf,
        'p_obj_onset': obj_p, 'p_obj_onset_shuf': obj_p_shuf,
        'p_reward': reward_p, 'p_reward_onset_shuf': reward_p_shuf, 
        'p_sound_onset': sound_onset_p,  'p_sound_onset_shuf': sound_onset_p_shuf
        })))
    
    # Save to disk
    stats_df.to_csv(join(path_dict['save_path'], 'significant_neurons.csv'), index=False)
        
           
        
    
    
