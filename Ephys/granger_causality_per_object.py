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
from sklearn.utils import shuffle as sklearn_shuffle
from joblib import Parallel, delayed
from msvr_functions import (paths, load_multiple_probes, load_subjects, load_objects,
                            calculate_peths)

# Settings
T_BEFORE = 2  # s
T_AFTER = 1
BIN_SIZE = 0.05
SMOOTHING = 0.1
MIN_NEURONS = 10
MIN_TRIALS = 30
N_CORES = -1
MAX_LAG = 0.5  # s
N_SHUFFLES = 500

# Initialize
clf = LogisticRegression(solver='liblinear')
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))


# Functions for parallelization
def decode_context(X, y, clf, shuffle=False):
    
    decoding_probs = np.empty((X.shape[0], X.shape[2]))  # shape: (n_trials, time)
    
    # Decode context per timebin
    for tb in range(X.shape[2]):
        
        X_t = X[:, :, tb]  # shape: (n_trials, n_neurons)
        if shuffle:
            y = sklearn_shuffle(y)        

        for tr in range(X.shape[0]):  # loop over trials
        
            # Leave-one-out
            X_train = np.delete(X_t, tr, axis=0)
            y_train = np.delete(y, tr)
            X_test = X_t[tr].reshape(1, -1)
            y_test = y[tr]

            # Train logistic regression 
            clf.fit(X_train, y_train)

            # Predict probability for the correct class
            prob = clf.predict_proba(X_test)[0, y_test]
            decoding_probs[tr, tb] = prob                    
    
    return decoding_probs


def granger_causality(trial_data):
                
    # Region 1 to region 2
    model = VAR(trial_data)
    res = model.fit(maxlags=int(MAX_LAG / BIN_SIZE))
    f_test = res.test_causality('region1', 'region2', kind='f')
    f_12 = f_test.test_statistic
    
    # Region 2 to region 1
    f_test = res.test_causality('region2', 'region1', kind='f')
    f_21 = f_test.test_statistic

    return f_12, f_21


def granger_shuffled(i_shuf, prob_shuf, trial, this_obj, region1, region2):
    trial_data = pd.DataFrame({'region1': prob_shuf[this_obj][region1][trial, :, i_shuf],
                               'region2': prob_shuf[this_obj][region2][trial, :, i_shuf]})  
    # Region 1 to region 2
    model = VAR(trial_data)
    res = model.fit(maxlags=int(MAX_LAG / BIN_SIZE))
    f_test = res.test_causality('region1', 'region2', kind='f')
    f_12 = f_test.test_statistic
    
    # Region 2 to region 1
    f_test = res.test_causality('region2', 'region1', kind='f')
    f_21 = f_test.test_statistic
    
    return f_12, f_21    


# %% Loop over recordings
granger_df = pd.DataFrame()
for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'\n{subject} {date} ({i} of {rec.shape[0]})')

    # Load in data
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_multiple_probes(session_path)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)
    
    if trials.shape[0] < MIN_TRIALS:
        continue
    
    # %% Loop over regions
    
    # Get list of all regions and which probe they were recorded on
    regions, region_probes = [], []
    for p, probe in enumerate(spikes.keys()):
        regions.append(np.unique(clusters[probe]['region']))
        region_probes.append([probe] * np.unique(clusters[probe]['region']).shape[0])
    regions = np.concatenate(regions)
    region_probes = np.concatenate(region_probes)
    
    prob, prob['object1'], prob['object2'], prob['object3'] = dict(), dict(), dict(), dict()
    prob_shuf, prob_shuf['object1'], prob_shuf['object2'], prob_shuf['object3'] = dict(), dict(), dict(), dict()
    for r, (region, probe) in enumerate(zip(regions, region_probes)):
        if region == 'root':
            continue
        print(f'Decoding {region}')
        
        # Get region neurons
        region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
        if region_neurons.shape[0] < MIN_NEURONS:
            continue
        
        # Loop over objects
        for this_obj in [1, 2, 3]:
        
            # Sound A versus sound B 
            _, soundA = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], region_neurons, 
                all_obj_df.loc[(all_obj_df['object'] == this_obj) & (all_obj_df['sound'] == 1), 'times'].values,
                T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
                )
            _, soundB = calculate_peths(
                spikes[probe]['times'], spikes[probe]['clusters'], region_neurons, 
                all_obj_df.loc[(all_obj_df['object'] == this_obj) & (all_obj_df['sound'] == 2), 'times'].values,
                T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
                )
            
            # Decode context per timebin and get decoding probabilities
            X = np.concatenate([soundA, soundB], axis=0)  # shape: (trials, neurons, time)
            y = np.concatenate([np.zeros(soundA.shape[0]), np.ones(soundB.shape[0])]).astype(int)
            prob[f'object{this_obj}'][region] = decode_context(X, y, clf)
            
            # Do the decoding while shuffling the context ids, use parallel processing for shuffling
            results = Parallel(n_jobs=N_CORES)(delayed(decode_context)(X, y, clf, shuffle=True)
                                               for t in range(N_SHUFFLES))
            prob_shuf[f'object{this_obj}'][region] = np.dstack(results) # shape: (trials, time, shuffles)
            
        
    # Do Granger causality for all region pairs
    print('Run Granger causality..')
    for this_obj in ['object1', 'object2', 'object3']:
        for region1, region2 in combinations(prob[this_obj].keys(), 2):
            
            # Do Granger causality per trial
            n_trials = prob[this_obj][region1].shape[0]
            reg1_reg2, reg2_reg1 = np.empty(n_trials), np.empty(n_trials)
            reg1_reg2_p, reg2_reg1_p = np.empty(n_trials), np.empty(n_trials) 
            for trial in range(n_trials):
                
                # Granger causality
                trial_data = pd.DataFrame({'region1': prob[this_obj][region1][trial],
                                           'region2': prob[this_obj][region2][trial]})  
                reg1_reg2[trial], reg2_reg1[trial] = granger_causality(trial_data)
            
                # Shuffles
                results = Parallel(n_jobs=N_CORES)(delayed(granger_shuffled)(
                    i_shuf, prob_shuf, trial, this_obj, region1, region2) for i_shuf in range(N_SHUFFLES))
                reg1_reg2_shuf = np.array([i[0] for i in results])
                reg2_reg1_shuf = np.array([i[1] for i in results])
  
                # Get p-value
                z = (reg1_reg2[trial] - np.mean(reg1_reg2_shuf)) / np.std(reg1_reg2_shuf)
                reg1_reg2_p[trial] = 2 * (1 - stats.norm.cdf(abs(z)))
                z = (reg2_reg1[trial] - np.mean(reg2_reg1_shuf)) / np.std(reg2_reg1_shuf)
                reg2_reg1_p[trial] = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # Get p-value over trials by doing a binomial test
            p_reg1_reg2 = stats.binomtest(np.sum(reg1_reg2_p < 0.05), n_trials, 0.05).pvalue
            p_reg2_reg1 = stats.binomtest(np.sum(reg2_reg1_p < 0.05), n_trials, 0.05).pvalue
            
            # Add to dataframe
            granger_df = pd.concat((granger_df, pd.DataFrame(data={
                'region_pair': [f'{region1} → {region2}', f'{region2} → {region1}'],
                'region1': [region1, region2], 'region2': [region2, region1],
                'object': this_obj,
                'p_value': [p_reg1_reg2, p_reg2_reg1],
                'f_stat': [np.median(reg1_reg2), np.median(reg2_reg1)],
                'subject': subject, 'date': date})))
        
    # Save to disk
    granger_df.to_csv(join(path_dict['save_path'], 'granger_causality_objects.csv'), index=False)
    