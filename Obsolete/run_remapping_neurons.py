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
PRE_OBJ = 450
ALPHA = 0.05
OVERWRITE = True
N_CORES = -4
MIN_SPEED = 20  # mm/s

# Initialize
path_dict = paths()
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# Load in previous data
if OVERWRITE:
    stats_df = pd.DataFrame()
else:
    stats_df = pd.read_csv(join(path_dict['save_path'], 'remapping_neurons.csv'),
                           dtype={'subject': str, 'date': str})
    merged = rec.merge(stats_df, on=['subject', 'date', 'probe'], how='left', indicator=True)
    rec = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    

# %% Function for parallel processing

def run_zetatest2(neuron_id, event_pos1, event_pos2, d_before, d_after, do_shuffle=False):
    these_spikes = spikes['distances'][spikes['clusters'] == neuron_id].astype(float)
    if do_shuffle:
        all_shuffled_events = shuffle(np.concatenate((event_pos1, event_pos2)))
        event_pos1 = all_shuffled_events[:event_pos1.shape[0]]
        event_pos2 = all_shuffled_events[event_pos2.shape[0]:]
    try:
        p_value, zeta_dict = zetatest2(these_spikes, event_pos1 - d_before, these_spikes,
                                       event_pos2 - d_before, max_duration=(d_before + d_after))
        zeta_score = zeta_dict['zeta_deviation']
    except Exception:
        p_value = np.nan    
        zeta_score = np.nan        
    return p_value, zeta_score


# %%
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    if subject not in subjects.loc[subjects['Far'] == 1, 'SubjectID'].values:
        continue
    print(f'\nStarting {subject} {date} {probe} [{i} of {rec.shape[0]}]\n')
    
    # Load in data
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe, histology=True, only_good=True,
                                                  min_fr=0.1)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)      
    
    # Set a speed threshold
    spikes['distances'] = spikes['distances'][spikes['speeds'] >= MIN_SPEED]
    spikes['clusters'] = spikes['clusters'][spikes['speeds'] >= MIN_SPEED]

    # Get positions of objects 
    obj1_goal = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 1), 'distances'].values
    obj1_nog = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 0), 'distances'].values
    obj2_goal = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 1), 'distances'].values
    obj2_nog = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 0), 'distances'].values

    # Run ZETA test on spike positions for obj1 (goal/non-goal) and obj2 (goal/non-goal)
    results = Parallel(n_jobs=N_CORES, verbose=1)(delayed(run_zetatest2)(
        neuron_id, obj1_goal.astype(float), obj1_nog.astype(float),
        d_before=PRE_OBJ, d_after=0, do_shuffle=False) for neuron_id in clusters['cluster_id'])
    p_values_1 = np.array([r[0] for r in results])

    results = Parallel(n_jobs=N_CORES, verbose=1)(delayed(run_zetatest2)(
        neuron_id, obj2_goal.astype(float), obj2_nog.astype(float),
        d_before=PRE_OBJ, d_after=0, do_shuffle=False) for neuron_id in clusters['cluster_id'])
    p_values_2 = np.array([r[0] for r in results])

    # Add to dataframe
    temp_df = pd.DataFrame({
        'subject': subject,
        'date': date,
        'probe': probe,
        'neuron_id': clusters['cluster_id'],
        'region': clusters['region'],
        'acronym': clusters['acronym'],
        'p_obj1': p_values_1,
        'p_obj2': p_values_2
    })
    stats_df = pd.concat([stats_df, temp_df], ignore_index=True)
    stats_df.to_csv(join(path_dict['save_path'], 'remapping_neurons.csv'), index=False)







        