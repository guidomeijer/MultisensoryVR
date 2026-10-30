# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:45:16 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from joblib import Parallel, delayed
from latenzy import latenzy, latenzy2
from msvr_functions import paths, load_neural_data, load_subjects, load_objects

# Settings
OVERWRITE = True
N_CPUS = 4

# Initialize
path_dict = paths()
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

# Load in previous data
if OVERWRITE:
    latency_df = pd.DataFrame()
else:
    latency_df = pd.read_csv(join(path_dict['save_path'], 'latency_neurons.csv'))
    latency_df[['subject', 'date', 'probe']] = latency_df[['subject', 'date', 'probe']].astype(str)
    merged = rec.merge(latency_df, on=['subject', 'date', 'probe'], how='left', indicator=True)
    rec = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

# %%
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'Starting {subject} {date} {probe} [{i} of {rec.shape[0]}]\n')
    
    # Load in data
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)      
     
    # Get event times
    goal_times = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 1), 'times'].values
    no_goal_times = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 0), 'times'].values

    # Get latency
    results = Parallel(n_jobs=N_CPUS)(
        delayed(latenzy2)(spikes['times'][np.isin(spikes['clusters'], n)], goal_times,
                          spikes['times'][np.isin(spikes['clusters'], n)], no_goal_times, use_dur=2)
        for n in clusters['cluster_id'])
    latency = np.array([i[0] for i in results])
        
    # Add to dataframe
    latency_df = pd.concat((latency_df, pd.DataFrame(data={
        'subject': subject, 'date': date, 'probe': probe, 'neuron_id': clusters['cluster_id'],
        'region': clusters['region'], 'allen_acronym': clusters['acronym'],
        'latency': latency
        })))
    
    # Save to disk
    latency_df.to_csv(join(path_dict['save_path'], 'latency_goal_nogoal_obj1_neurons.csv'), index=False)
        
           
        
    
    
