# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
import random
from itertools import combinations
from statsmodels.tsa.api import VAR
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
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
N_CORES = 12
MAX_LAG = 0.5  # s
N_PSEUDO = 500

# Initialize
#clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=500))
clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=1))
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))


# Functions for parallelization
def decode_context(X, y, clf):
    """
    Decodes object identity per timebin using efficient leave-one-out cross-validation.
    
    Args:
        X (np.array): Data of shape (n_trials, n_neurons, n_timebins)
        y (np.array): Labels of shape (n_trials,)
        clf: Scikit-learn classifier instance.
        
    Returns:
        np.array: Decoding probabilities of shape (n_trials, n_timebins)
    """
    decoding_probs = np.empty((X.shape[0], X.shape[2]))
    loo = LeaveOneOut()
    
    # Get unique classes to correctly index probabilities
    classes = np.unique(y)

    for tb in range(X.shape[2]):
        X_t = X[:, :, tb] # shape: (n_trials, n_neurons)
        
        # Get probabilities for all classes using cross-validation
        all_probs = cross_val_predict(clf, X_t, y, cv=loo, method='predict_proba', n_jobs=N_CORES)
        
        # Select the probability corresponding to the *true* label for each trial
        true_class_indices = np.searchsorted(classes, y)
        decoding_probs[:, tb] = all_probs[np.arange(len(y)), true_class_indices]
        
    return decoding_probs


def granger_causality(trial_data):
    """
    Calculates Granger causality between two time series in a NumPy array.
    Assumes column 0 is the first region and column 1 is the second.
    """
    model = VAR(trial_data)
    res = model.fit(maxlags=int(MAX_LAG / BIN_SIZE))
    
    # Test causality from Region 1 -> Region 2
    # This tests if the lagged values of column 0 help predict column 1.
    # caused = 1 (region2), causing = 0 (region1)
    f_test = res.test_causality(caused=1, causing=[0], kind='f')
    f_12 = f_test.test_statistic
    
    # Test causality from Region 2 -> Region 1
    # This tests if the lagged values of column 1 help predict column 0.
    # caused = 0 (region1), causing = 1 (region2)
    f_test = res.test_causality(caused=0, causing=[1], kind='f')
    f_21 = f_test.test_statistic

    return f_12, f_21   


def process_trial_granger(trial_idx, prob_dict, r1, r2, n_shuffles):
    """
    Performs Granger causality using a time-shifting surrogate method.
    """
    # ... (Get real_ts_r1, real_ts_r2, check for NaNs, etc.) ...
    real_ts_r1 = prob_dict[r1][trial_idx]
    real_ts_r2 = prob_dict[r2][trial_idx]
    n_timepoints = len(real_ts_r1)

    if not np.all(np.isfinite(real_ts_r1)) or not np.all(np.isfinite(real_ts_r2)):
        return np.nan, np.nan, np.nan, np.nan
        
    # 1. Calculate REAL Granger causality
    real_trial_data = np.vstack([real_ts_r1, real_ts_r2]).T
    f_12, f_21 = granger_causality(real_trial_data)

    # 2. Generate NULL distribution by circularly shifting one time series
    reg1_reg2_shuf = np.empty(n_shuffles)
    reg2_reg1_shuf = np.empty(n_shuffles)

    for i in range(n_shuffles):
        # Generate a random shift. It must not be 0.
        # Ensure shift is > max_lag to break short-term relationships.
        shift = np.random.randint(MAX_LAG + 1, n_timepoints - (MAX_LAG + 1))
        
        # Shift the SECOND time series
        shifted_ts_r2 = np.roll(real_ts_r2, shift)
        
        # Now create the surrogate pair using the original r1 and the shifted r2
        surrogate_data = np.vstack([real_ts_r1, shifted_ts_r2]).T
        reg1_reg2_shuf[i], reg2_reg1_shuf[i] = granger_causality(surrogate_data)

    # 3. Calculate p-values
    p_12 = (np.sum(reg1_reg2_shuf >= f_12) + 1) / (n_shuffles + 1)
    p_21 = (np.sum(reg2_reg1_shuf >= f_21) + 1) / (n_shuffles + 1)
    
    return f_12, f_21, p_12, p_21


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
        
    # Do Granger causality for all region pairs
    for region1, region2 in combinations(prob['object1'].keys(), 2):
        print(f'Running Granger causality for {region1} → {region2}')
        for this_obj in [1, 2, 3]:
            n_trials = prob[f'object{this_obj}'][region1].shape[0]
            
            # Use joblib to parallelize the loop over trials
            # This will run 'process_trial_granger' for each trial on a different core
            results = Parallel(n_jobs=N_CORES)(delayed(process_trial_granger)(
                trial, prob[f'object{this_obj}'], region1, region2, N_PSEUDO) for trial in range(n_trials))
        
            # Unpack the list of tuples returned by Parallel into separate lists/tuples
            reg1_reg2, reg2_reg1, reg1_reg2_p, reg2_reg1_p = zip(*results)
        
            # Convert the results back to numpy arrays for the subsequent analysis
            reg1_reg2 = np.array(reg1_reg2)
            reg2_reg1 = np.array(reg2_reg1)
            reg1_reg2_p = np.array(reg1_reg2_p)
            reg2_reg1_p = np.array(reg2_reg1_p)
            
            # Handle any NaN trials that may have occurred
            # For example, if you need to perform the binomial test on valid trials only:
            valid_trials = ~np.isnan(reg1_reg2_p)
            p_reg1_reg2 = stats.binomtest(np.sum(reg1_reg2_p[valid_trials] < 0.05), 
                                          np.sum(valid_trials), 0.05).pvalue
                                          
            valid_trials = ~np.isnan(reg2_reg1_p)
            p_reg2_reg1 = stats.binomtest(np.sum(reg2_reg1_p[valid_trials] < 0.05), 
                                          np.sum(valid_trials), 0.05).pvalue
                
            # Add to dataframe
            granger_df = pd.concat((granger_df, pd.DataFrame(data={
                'region_pair': [f'{region1} → {region2}', f'{region2} → {region1}'],
                'region1': [region1, region2], 'region2': [region2, region1],
                'object': f'object{this_obj}',
                'p_value': [p_reg1_reg2, p_reg2_reg1],
                'f_stat': [np.mean(reg1_reg2), np.mean(reg2_reg1)],
                'subject': subject, 'date': date})))
        
    # Save to disk
    granger_df.to_csv(join(path_dict['save_path'], 'granger_causality_context.csv'), index=False)
    