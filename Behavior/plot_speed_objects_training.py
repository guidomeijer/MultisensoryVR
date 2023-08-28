# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:57:35 2023

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
import os
from os.path import join, isfile
import pandas as pd
from msvr_functions import load_subjects, paths, peri_event_trace, figure_style

SPEED_BIN_SIZE = 0.1  # s

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths()

# Set figure style
colors, dpi = figure_style()

# Loop over subjects
for i, subject in enumerate(subjects['SubjectID']):
    
    # Loop over sessions
    sessions = os.listdir(join(path_dict['server_path'], 'Subjects', subject))
    
    # Create figure for all sessions
    f, axs = plt.subplots(int(np.ceil(len(sessions)/5)), 5, figsize=(7, 1.75*np.ceil(len(sessions) / 5)),
                          dpi=dpi)
    if len(sessions) > 5:
        axs = np.concatenate(axs)
    
    for j, ses in enumerate(sessions):
        if isfile(join(path_dict['server_path'], 'Subjects', subject, ses, 'trials.csv')):
            
            # Load in data
            trials = pd.read_csv(join(path_dict['server_path'], 'Subjects', subject, ses, 'trials.csv'),
                                 index_col=0)
            wheel_times = np.load(join(path_dict['server_path'], 'Subjects', subject, ses, 'wheel.times.npy'))
            wheel_speed = np.load(join(path_dict['server_path'], 'Subjects', subject, ses, 'wheel.speed.npy'))
            
            # Downsample wheel speed
            wheel_speed = wheel_speed[::50]
            wheel_times = wheel_times[::50]
            
            # Plot
            all_obj_enters = np.concatenate((trials['enterObj1'], trials['enterObj2'], trials['enterObj3']))
            all_obj_enters = np.sort(all_obj_enters[~np.isnan(all_obj_enters)])
            peri_event_trace(wheel_speed, wheel_times, all_obj_enters,
                             event_ids=np.ones(all_obj_enters.shape),
                             t_before=1, t_after=6, ax=axs[j])
            axs[j].set(xlabel='Time (s)', ylabel='Speed (unitys/s)', xticks=np.arange(-1, 7))
            #sns.despine(trim=True)
            plt.tight_layout()
            
            