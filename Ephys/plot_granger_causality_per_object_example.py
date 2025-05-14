# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from itertools import combinations
from statsmodels.tsa.api import VAR
from scipy import stats
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from msvr_functions import (paths, load_multiple_probes, load_subjects, load_objects,
                            calculate_peths, figure_style)
colors, dpi = figure_style()

# Settings
T_BEFORE = 1  # s
T_AFTER = 1
BIN_SIZE = 0.05
SMOOTHING = 0.1
MIN_NEURONS = 10
MIN_TRIALS = 30
N_CORES = -1
MAX_LAG = 0.5  # s
SUBJECT = '459601'
SESSION = '20240411'

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))


# Functions for parallelization
def get_decoding_probabilities(t, X, y):
    
    X_t = X[:, :, t]  # shape: (n_trials, n_neurons)
    decoding_probs = np.empty(X.shape[0])  # shape: (n_trials)

    for i in range(X.shape[0]):  # loop over trials
        # Leave-one-out
        X_train = np.delete(X_t, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X_t[i].reshape(1, -1)
        y_test = y[i]

        # Train logistic regression (use liblinear for small data)
        clf = LogisticRegression(solver='liblinear')
        clf.fit(X_train, y_train)

        # Predict probability for the correct class
        prob = clf.predict_proba(X_test)[0, y_test]
        decoding_probs[i] = prob

    return decoding_probs  # shape: (n_trials, n_timebins)


def granger_causality(trial, prob, this_obj, region1, region2):
                
    # Region 1 to region 2
    trial_data = pd.DataFrame({'region1': prob[this_obj][region1][trial],
                               'region2': prob[this_obj][region2][trial]})  
    model = VAR(trial_data)
    res = model.fit(maxlags=int(MAX_LAG / BIN_SIZE))
    f_test = res.test_causality('region1', 'region2', kind='f')
    p_12, f_12 = f_test.pvalue, f_test.test_statistic
    
    # Region 2 to region 1
    f_test = res.test_causality('region2', 'region1', kind='f')
    p_21, f_21 = f_test.pvalue, f_test.test_statistic

    return p_12, f_12, p_21, f_21


# Load in data
session_path = join(path_dict['local_data_path'], 'subjects', f'{SUBJECT}', f'{SESSION}')
spikes, clusters, channels = load_multiple_probes(session_path)
trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', SUBJECT, SESSION, 'trials.csv'))
all_obj_df = load_objects(SUBJECT, SESSION)


# %% Loop over regions

# Get list of all regions and which probe they were recorded on
regions, region_probes = [], []
for p, probe in enumerate(spikes.keys()):
    regions.append(np.unique(clusters[probe]['region']))
    region_probes.append([probe] * np.unique(clusters[probe]['region']).shape[0])
regions = np.concatenate(regions)
region_probes = np.concatenate(region_probes)

prob, prob['object1'], prob['object2'], prob['object3'] = dict(), dict(), dict(), dict()
for r, (region, probe) in enumerate(zip(regions, region_probes)):
    if region == 'root':
        continue
    print(f'Decoding {region}')
    
    # Get region neurons
    region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
    if region_neurons.shape[0] < MIN_NEURONS:
        continue
    
    # Sound A versus sound B at object 1
    peth, obj1_soundA = calculate_peths(
        spikes[probe]['times'], spikes[probe]['clusters'], region_neurons, 
        all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['sound'] == 1), 'times'].values,
        T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
        )
    _, obj1_soundB = calculate_peths(
        spikes[probe]['times'], spikes[probe]['clusters'], region_neurons, 
        all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['sound'] == 2), 'times'].values,
        T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
        )
    
    # Sound A versus sound B at object 2
    _, obj2_soundA = calculate_peths(
        spikes[probe]['times'], spikes[probe]['clusters'], region_neurons, 
        all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['sound'] == 1), 'times'].values,
        T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
        )
    _, obj2_soundB = calculate_peths(
        spikes[probe]['times'], spikes[probe]['clusters'], region_neurons, 
        all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['sound'] == 2), 'times'].values,
        T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
        )
    
    # Sound A versus sound B at object 3
    _, obj3_soundA = calculate_peths(
        spikes[probe]['times'], spikes[probe]['clusters'], region_neurons, 
        all_obj_df.loc[(all_obj_df['object'] == 3) & (all_obj_df['sound'] == 1), 'times'].values,
        T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
        )
    _, obj3_soundB = calculate_peths(
        spikes[probe]['times'], spikes[probe]['clusters'], region_neurons, 
        all_obj_df.loc[(all_obj_df['object'] == 3) & (all_obj_df['sound'] == 2), 'times'].values,
        T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
        )
    
    # Prepare data for decoding
    X = dict()
    X['object1'] = np.concatenate([obj1_soundA, obj1_soundB], axis=0)  # shape: (n_trials_total, n_neurons, n_timebins)
    X['object2'] = np.concatenate([obj2_soundA, obj2_soundB], axis=0)  
    X['object3'] = np.concatenate([obj3_soundA, obj3_soundB], axis=0)
    y = np.concatenate([np.zeros(obj1_soundA.shape[0]), np.ones(obj1_soundB.shape[0])]).astype(int)
    
    # Run decoding in parallel
    for this_obj in ['object1', 'object2', 'object3']:
        results = Parallel(n_jobs=N_CORES)(delayed(get_decoding_probabilities)(t, X[this_obj], y)
                                           for t in range(X[this_obj].shape[2]))
        prob[this_obj][region] = np.vstack(results).T  # shape: (trials x timebins)

# Do Granger causality for all region pairs
print('Run Granger causality..')
for this_obj in ['object1', 'object2', 'object3']:
    for region1, region2 in combinations(prob[this_obj].keys(), 2):
        
        # Do Granger causality per trial
        n_trials = prob[this_obj][region1].shape[0]
        results = Parallel(n_jobs=N_CORES)(delayed(granger_causality)(
            trial, prob, this_obj, region1, region2) for trial in range(n_trials))
        p_12, f_12 = np.array([i[0] for i in results]), np.array([i[1] for i in results])
        p_21, f_21 = np.array([i[2] for i in results]), np.array([i[3] for i in results])
        
        # Get p-value over trials by doing a binomial test
        p_12_binom = stats.binomtest(np.sum(p_12 < 0.05), n_trials, 0.05).pvalue
        p_21_binom = stats.binomtest(np.sum(p_21 < 0.05), n_trials, 0.05).pvalue
        
# %% Plot

plot_trial = 20
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.9), dpi=dpi)
ax1.plot(peth['tscale'], prob['object2']['dCA1'][plot_trial, :], label='dCA1', color=colors['dCA1'])
ax1.plot(peth['tscale'], prob['object2']['ENT'][plot_trial, :], label='ENT', color=colors['ENT'])
ax1.set(xlabel='Time from object entry (s)', ylabel='Decoding probability (%)', yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        title=f'ENT → dCA1: {np.round(f_12[plot_trial], 1)}\nENT ← dCA1: {np.round(f_21[plot_trial], 1)}')
ax1.legend()

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(path_dict['google_drive_fig_path'], 'granger_example.jpg'), dpi=600)
   