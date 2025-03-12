# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, isdir
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.plot import peri_event_time_histogram
from msvr_functions import (paths, peri_multiple_events_time_histogram, calculate_peths,
                            load_neural_data, figure_style, load_subjects)

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
T_BEFORE = 1  # s
T_AFTER = 2
BIN_SIZE = 0.05
SMOOTHING = 0.1
MIN_SPIKES = 10

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
colors, dpi = figure_style()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE, histology=True, only_good=True)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))

# Get reward contingencies
sound1_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'Sound1Obj'].values[0]
sound2_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'Sound2Obj'].values[0]
control_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'ControlObject'].values[0]

obj1_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 1)[0][0] + 1
obj2_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 2)[0][0] + 1
obj3_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 3)[0][0] + 1

# Prepare trial data
obj1_df = pd.DataFrame(data={'times': trials['enterObj1'], 'object': 1, 'sound': trials['soundId'],
                             'goal': (trials['soundId'] == obj1_goal_sound).astype(int)})
obj2_df = pd.DataFrame(data={'times': trials['enterObj2'], 'object': 2, 'sound': trials['soundId'],
                             'goal': (trials['soundId'] == obj2_goal_sound).astype(int)})
obj3_df = pd.DataFrame(data={'times': trials['enterObj3'], 'object': 3, 'sound': trials['soundId'],
                             'goal': (trials['soundId'] == obj3_goal_sound).astype(int)})
all_obj_df = pd.concat((obj1_df, obj2_df, obj3_df))
all_obj_df = all_obj_df.sort_values(by='times').reset_index(drop=True)

# Prepare data for GLM
X = all_obj_df[['object', 'sound', 'goal']]  
X = sm.add_constant(X)  # Add a constant for the intercept

# %% Loop over time bins
glm_df = pd.DataFrame()

peths, binned_spikes = calculate_peths(spikes['times'], spikes['clusters'],
                                       np.unique(spikes['clusters']), all_obj_df['times'],
                                       pre_time=T_BEFORE, post_time=T_AFTER,
                                       bin_size=BIN_SIZE, smoothing=SMOOTHING)
t_centers = peths['tscale']

for i, bin_center in enumerate(t_centers):
    print(f'Timebin {np.round(bin_center, 2)} ({i} of {len(t_centers)})')
       
    # Loop over neurons
    for n, neuron_id in enumerate(np.unique(spikes['clusters'])):
        
        if np.sum(binned_spikes[:, n, i]) < 0.5:
            
             glm_df = pd.concat((glm_df, pd.DataFrame(index=[glm_df.shape[0]], data={
                 'neuron_id': neuron_id,
                 'allen_acronym': clusters['acronym'][clusters['cluster_id'] == neuron_id][0],
                 'time': bin_center,
                 'coef_object': np.nan,
                 'coef_sound': np.nan,
                 'coef_goal': np.nan,
                 'p_object': np.nan,
                 'p_sound': np.nan,
                 'p_goal': np.nan
                 })))
         
        else:
            
            try:
                # Fit the GLM model with a Poisson regression
                glm_model = sm.GLM(binned_spikes[:, n, i], X, family=sm.families.Poisson()).fit()
            except Exception:
                glm_df = pd.concat((glm_df, pd.DataFrame(index=[glm_df.shape[0]], data={
                    'neuron_id': neuron_id,
                    'allen_acronym': clusters['acronym'][clusters['cluster_id'] == neuron_id][0],
                    'time': bin_center,
                    'coef_object': np.nan,
                    'coef_sound': np.nan,
                    'coef_goal': np.nan,
                    'p_object': np.nan,
                    'p_sound': np.nan,
                    'p_goal': np.nan
                    })))
                
            glm_df = pd.concat((glm_df, pd.DataFrame(index=[glm_df.shape[0]], data={
                'neuron_id': neuron_id,
                'allen_acronym': clusters['acronym'][clusters['cluster_id'] == neuron_id][0],
                'time': bin_center,
                'coef_object': glm_model.params['object'],
                'coef_sound': glm_model.params['sound'],
                'coef_goal': glm_model.params['goal'], 
                'p_object': glm_model.pvalues['object'],
                'p_sound': glm_model.pvalues['sound'],
                'p_goal': glm_model.pvalues['goal']
                })))
                        
# Save to disk
glm_df.to_csv(join(path_dict['save_path'], 'glm_results.csv'), index=False)

"""            
# %% Plot
for n, neuron_id in enumerate(np.unique(glm_df['neuron_id'])):
    
    neuron_id = 311
    this_df = glm_df[glm_df['neuron_id'] == neuron_id]

    f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
    ax1.plot(this_df['time'], np.abs(this_df['coef_goal']))
    ax1.plot(this_df['time'], np.abs(this_df['coef_object']))
    ax1.plot(this_df['time'], np.abs(this_df['coef_sound']))
"""


