# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:38:04 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from msvr_functions import paths, load_neural_data, figure_style

colors, dpi = figure_style()
path_dict = paths()
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)

waveform_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'Session {i} of {rec.shape[0]}')
    
    # Load in waveform metrics
    sort_path = path_dict['server_path'] / 'Subjects' / subject / date / probe / 'sorting'
    ses_path = path_dict['local_data_path'] / 'Subjects' / subject / date 
    waveform_metrics = pd.read_csv(sort_path / 'extensions' / 'template_metrics' / 'metrics.csv')
    waveform_metrics = waveform_metrics.rename(columns={'Unnamed: 0': 'neuron_id'})
    
    # Load in region
    spikes, clusters, channels = load_neural_data(ses_path, probe, only_good=False, min_fr=0)
    waveform_metrics['region'] = clusters['region']
    waveform_metrics['firing_rate'] = clusters['firing_rate']
    waveform_metrics['good'] = ((clusters['ibl_label'] == 1) | (clusters['ml_label'] == 1)).astype(int)
        
    # Add to dataframe
    waveform_metrics['subject'] = subject
    waveform_metrics['date'] = date
    waveform_metrics['probe'] = probe
    waveform_df = pd.concat((waveform_df, waveform_metrics), ignore_index=True)

# Save result
waveform_df.to_csv(path_dict['save_path'] / 'waveform_metrics.csv', index=False)

