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
OVERWRITE = True

# Initialize
path_dict = paths()
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# Load in previous data
if OVERWRITE:
    speed_df = pd.DataFrame()
else:
    speed_df = pd.read_csv(join(path_dict['save_path'], 'speed_neurons.csv'))
    speed_df[['subject', 'date', 'probe']] = speed_df[['subject', 'date', 'probe']].astype(str)
    merged = rec.merge(speed_df, on=['subject', 'date', 'probe'], how='left', indicator=True)
    rec = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    
# %%
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\nStarting {subject} {date} {probe}..')
    
    # Load in data
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe)
    wheel_speed = np.load(join(session_path, 'continuous.wheelSpeed.npy'))
    timestamps = np.load(join(session_path, 'continuous.times.npy'))
    
    
    
    
   
   
    # Add to dataframe
    speed_df = pd.concat((speed_df, pd.DataFrame(data={
        'subject': subject, 'date': date, 'probe': probe, 'neuron_id': clusters['cluster_id'],
        'region': clusters['region'], 'allen_acronym': clusters['acronym'],
        'x': clusters['x'], 'y': clusters['y'], 'z': clusters['z'],
        'sig_context_obj1': goal1_p < ALPHA, 'sig_context_obj2': goal2_p < ALPHA,
        'sig_obj_onset': obj_p < ALPHA, 'sig_control': control_sound_p < ALPHA,
        'sig_reward': reward_p < ALPHA, 'sig_omission': omission_p < ALPHA,
        'sig_sound_onset': sound_onset_p < ALPHA, 'p_sound_onset': sound_onset_p,
        'p_goal_obj1': goal1_p, 'p_goal_obj2': goal2_p,
        'p_control_sound': control_sound_p, 'p_obj_onset': obj_p,
        'p_reward': reward_p, 'p_omission': omission_p
        })))
        
    # Save to disk
    speed_df.to_csv(join(path_dict['save_path'], 'speed_neurons.csv'), index=False)
        
           
        
    
    
