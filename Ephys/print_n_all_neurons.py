# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 08:44:15 2025 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from scipy import stats
from msvr_functions import paths, load_neural_data, load_subjects
path_dict = paths()

rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
n_neurons = []
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\nStarting {subject} {date} {probe}..')
    
    # Load in data
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe,
                                                  histology=False,
                                                  only_good=False,
                                                  min_fr=0)
    n_neurons.append(len(clusters['cluster_id']))

print(f'{np.sum(n_neurons)} total number of detected units')
print(f'{np.mean(n_neurons)} +- {stats.sem(n_neurons)} (mean += sem) per probe')