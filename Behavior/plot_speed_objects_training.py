# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:57:35 2023

By Guido Meijer
"""

import os
from os.path import join, isfile
import pandas as pd
from msvr_functions import load_subjects, paths

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths()

# Loop over subjects
for i, subject in enumerate(subjects['SubjectID']):
    
    # Loop over sessions
    sessions = os.listdir(join(path_dict['server_path'], 'Subjects', subject))
    for j, ses in enumerate(sessions):
        if isfile(join(path_dict['server_path'], 'Subjects', subject, ses, 'trials.csv')):
            trials = pd.read_csv(join(path_dict['server_path'], 'Subjects', subject, ses, 'trials.csv'))
            asd