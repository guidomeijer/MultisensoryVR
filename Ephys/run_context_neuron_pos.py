# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:38:48 2025

By Guido Meijer
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style, load_neural_data


# Load in data
path_dict = paths()
neurons_df = pd.read_csv(path_dict['save_path'] / 'significant_neurons.csv')
neurons_df['date'] = neurons_df['date'].astype(str)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)

neuron_pos_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\nStarting {subject} {date} {probe} [{i} of {rec.shape[0]}]\n')
    
    # Load in data
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
    spikes, clusters, channels = load_neural_data(session_path, probe, histology=True, only_good=True,
                                                  min_fr=0.1)
    trials = pd.read_csv(path_dict['local_data_path'] / 'Subjects' / subject / date / 'trials.csv')
    these_neurons = neurons_df[(neurons_df['date'] == date) & (neurons_df['probe'] == probe)]
    
    # Add to dataframe
    neuron_pos_df = pd.concat((neuron_pos_df, pd.DataFrame(data={
        'p_context': these_neurons['p_context_obj2'],
        'x': clusters['x'], 'y': clusters['y'], 'z': clusters['z'],
        'neuron_id': clusters['cluster_id'],
        'region': clusters['region'], 'acronym': clusters['acronym'],
        'subject': subject, 'date': date, 'probe': probe
        })))
    
    # Save to disk
    neuron_pos_df.to_csv(path_dict['save_path'] / 'neuron_position.csv', index=False)