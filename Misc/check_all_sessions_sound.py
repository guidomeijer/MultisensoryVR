# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 10:00:19 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
from msvr_functions import paths, load_objects, load_subjects

# Initialize
path_dict = paths()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)

for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'\nStarting {subject} {date}..')
    
    # Load data
    all_obj_df = load_objects(subject, date)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', subject, date, 'trials.csv'))
    subjects = load_subjects()
    
    test_arr = all_obj_df.loc[(all_obj_df['sound'] == 1) & (all_obj_df['object'] == 1), 'goal'].values
    if not np.all(test_arr == test_arr[0]):
        print('Error')
        
    test_arr = all_obj_df.loc[(all_obj_df['sound'] == 2) & (all_obj_df['object'] == 1), 'goal'].values
    if not np.all(test_arr == test_arr[0]):
        print('Error')
        
    test_arr = all_obj_df.loc[(all_obj_df['sound'] == 1) & (all_obj_df['object'] == 2), 'goal'].values
    if not np.all(test_arr == test_arr[0]):
        print('Error')
        
    test_arr = all_obj_df.loc[(all_obj_df['sound'] == 2) & (all_obj_df['object'] == 2), 'goal'].values
    if not np.all(test_arr == test_arr[0]):
        print('Error')
    
    if np.sum(all_obj_df.loc[all_obj_df['object'] == 3, 'goal']) != 0:
        print('Error')
        
    print((trials['soundOnsetPos'] - trials['enterEnvPos']).values)
    
    