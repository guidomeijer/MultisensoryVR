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
from msvr_functions import (paths, peri_multiple_events_time_histogram, 
                            load_neural_data, figure_style, load_subjects)

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
T_BEFORE = 1  # s
T_AFTER = 2
BIN_SIZE = 0.1
SMOOTHING = 0.05
STEP_SIZE = 0.01
MIN_SPIKES = 10

# Create time array
t_centers = np.arange(-T_BEFORE + (BIN_SIZE/2), T_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Function to predict firing rates with a GLM
def fit_poisson_regression(X_train, y_train, X_test):
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    return model, model.predict(X_test)

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
X = all_obj_df[['object', 'sound', 'goal']]  # for GLM fit

# %% Loop over time bins
glm_df = pd.DataFrame()
for i, bin_center in enumerate(t_centers):
    print(f'Timebin {np.round(bin_center, 2)} ({i} of {len(t_centers)})')
    
    # Get spike counts per trial for all neurons during this time bin
    these_intervals = np.vstack((all_obj_df['times'] + (bin_center - (BIN_SIZE/2)),
                                 all_obj_df['times'] + (bin_center + (BIN_SIZE/2)))).T
    spike_counts, neuron_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'],
                                                        these_intervals)
    
    # Loop over neurons
    for n, neuron_id in enumerate(neuron_ids):
        if np.sum(spike_counts[n, :]) >= MIN_SPIKES:
            
            # Fit the model using cross-validation
            y = spike_counts[n, :]            
            pseudo_r2_scores = []
            p_values, coeffs = {col: [] for col in X.columns}, {col: [] for col in X.columns}
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
            
                # Fit the model and predict on the test set
                model, y_pred = fit_poisson_regression(X_train, y_train, X_test)
            
                # Calculate the null deviance and residual deviance
                null_model = sm.GLM(y_train, np.ones((len(y_train), 1)), family=sm.families.Poisson()).fit()
                null_deviance = null_model.deviance
                residual_deviance = model.deviance
            
                # Calculate pseudo R^2
                pseudo_r2 = 1 - (residual_deviance / null_deviance)
                pseudo_r2_scores.append(pseudo_r2)
                
                for col in X.columns:
                    p_values[col].append(model.pvalues[col])
                    coeffs[col].append(model.params[col])
                
            # Get means over folds
            p_object = np.mean(p_values['object'])
            p_sound = np.mean(p_values['sound'])
            p_goal = np.mean(p_values['goal'])
            pseudo_r2 = np.mean(pseudo_r2_scores)
            coef_object = np.mean(coeffs['object'])
            coef_sound = np.mean(coeffs['sound'])
            coef_goal = np.mean(coeffs['goal'])
            
            # Add to dataframe
            glm_df = pd.concat((glm_df, pd.DataFrame(index=[glm_df.shape[0]], data={
                'neuron_id': neuron_id, 'time': bin_center, 'pseudo_R2': pseudo_r2,
                'coef_object': coef_object, 'coef_sound': coef_sound, 'coef_goal': coef_goal, 
                'p_object': p_object, 'p_sound': p_sound, 'p_goal': p_goal,
                'allen_acronym': clusters['acronym'][clusters['cluster_id'] == neuron_id][0]})))
    
# %% Plot
for n, neuron_id in enumerate(np.unique(glm_df['neuron_id'])):
    this_df = glm_df[glm_df['neuron_id'] == neuron_id]
    if any(this_df['p_object'] < 0.05):
        f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.75), dpi=dpi)
        ax1.plot(this_df['time'], this_df['coef_goal'])
        ax1.plot(this_df['time'], this_df['coef_object'])
        ax1.plot(this_df['time'], this_df['coef_sound'])
        asd


