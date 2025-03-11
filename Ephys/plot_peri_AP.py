# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:20:19 2025 by Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style, combine_regions, load_neural_data
colors, dpi = figure_style()

# Load in data
path_dict = paths()
stats_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'Starting {subject} {date} {probe}..')
    
    # Load in data
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe)
    