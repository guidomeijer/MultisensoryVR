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
N_CORES = 18
MAX_LAG = 0.5  # s
N_PSEUDO = 500

# Initialize
# LogisticRegression is often significantly faster than RandomForest for this application
# clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=500))
clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=1))
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))


# Optimized Functions
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
        # Use n_jobs=N_CORES here as this runs outside the main parallel loop
        all_probs = cross_val_predict(clf, X_t, y, cv=loo, method='predict_proba', n_jobs=N_CORES)
        
        # Select the probability corresponding to the *true* label for each trial
        true_class_indices = np.searchsorted(classes, y)
        decoding_probs[:, tb] = all_probs[np.arange(len(y)), true_class_indices]
        
    return decoding_probs


def fast_bivariate_gc_f(target, source, lags):
    """
    Fast bivariate Granger Causality F-statistic calculation using OLS.
    Checks if source lags help predict target better than target lags alone.
    """
    n = len(target)
    if n <= (lags * 2 + 1):
        return np.nan

    # Prepare lagged matrices
    y_target = target[lags:]
    y_lags = np.column_stack([target[lags-1-i : n-1-i] for i in range(lags)])
    x_lags = np.column_stack([source[lags-1-i : n-1-i] for i in range(lags)])
    
    # Restricted model: target ~ constant + target_lags
    X_res = np.column_stack([np.ones(n - lags), y_lags])
    # Full model: target ~ constant + target_lags + source_lags
    X_full = np.column_stack([X_res, x_lags])

    try:
        # Solve OLS and get Residual Sum of Squares (RSS)
        _, rss_res, _, _ = np.linalg.lstsq(X_res, y_target, rcond=None)
        _, rss_full, _, _ = np.linalg.lstsq(X_full, y_target, rcond=None)
        
        if len(rss_res) == 0 or len(rss_full) == 0:
            return np.nan
            
        df_num = lags
        df_denom = n - (1 + 2 * lags)
        f_stat = ((rss_res[0] - rss_full[0]) / df_num) / (rss_full[0] / df_denom)
        return max(0, f_stat) # F-stat cannot be negative
    except:
        return np.nan


def compute_granger_trial_shuffling(residuals_r1, residuals_r2, max_lag_bins, n_shuffles=500):
    """
    Computes Granger Causality using trial shuffling on residuals with fast OLS.
    """
    n_trials = residuals_r1.shape[0]
    
    # Precompute all possible pairwise trial combinations (N^2)
    # This is significantly faster than computing them repeatedly during shuffles
    f12_matrix = np.empty((n_trials, n_trials))
    f21_matrix = np.empty((n_trials, n_trials))
    
    for i in range(n_trials):
        for j in range(n_trials):
            # f12: r1 -> r2. target is r2[i], source is r1[j]
            f12_matrix[i, j] = fast_bivariate_gc_f(residuals_r2[i], residuals_r1[j], max_lag_bins)
            # f21: r2 -> r1. target is r1[i], source is r2[j]
            f21_matrix[i, j] = fast_bivariate_gc_f(residuals_r1[i], residuals_r2[j], max_lag_bins)

    # 1. Real Granger Causality (diagonal of the matrices - matched trials)
    real_f12 = np.nanmean(np.diag(f12_matrix))
    real_f21 = np.nanmean(np.diag(f21_matrix))

    if np.isnan(real_f12) or np.isnan(real_f21):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # 2. Generate NULL Distribution by sampling indices from the precomputed matrices
    null_f12_dist = np.empty(n_shuffles)
    null_f21_dist = np.empty(n_shuffles)
    indices = np.arange(n_trials)
    
    for s in range(n_shuffles):
        shuf = np.random.permutation(n_trials)
        null_f12_dist[s] = np.nanmean(f12_matrix[shuf, indices])
        null_f21_dist[s] = np.nanmean(f21_matrix[indices, shuf])

    # 3. Calculate P-values
    p_12 = (np.sum(null_f12_dist >= real_f12) + 1) / (n_shuffles + 1)
    p_21 = (np.sum(null_f21_dist >= real_f21) + 1) / (n_shuffles + 1)
    null_f12_val = np.nanmedian(null_f12_dist)
    null_f21_val = np.nanmedian(null_f21_dist)

    return real_f12, null_f12_val, p_12, real_f21, null_f21_val, p_21


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
    
    residuals, residuals['object1'], residuals['object2'], residuals['object3'] = dict(), dict(), dict(), dict()
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
            
            # Get trial by trial decoding probabilities and subtract the mean to leave the residuals
            trial_probs = decode_context(X, y, clf)
            subtracted_probs = trial_probs - np.mean(trial_probs, axis=0)
            residuals[f'object{this_obj}'][region] = subtracted_probs
        
    # Loop over region pairs
    job_list = []
    max_lag_bins = int(MAX_LAG / BIN_SIZE)

    for region1, region2 in combinations(residuals['object1'].keys(), 2):
        for this_obj in [1, 2, 3]:
            job_list.append({
                'r1': region1, 'r2': region2,
                'obj': this_obj,
                'd1': residuals[f'object{this_obj}'][region1],
                'd2': residuals[f'object{this_obj}'][region2]
            })

    def run_pair_job(job):
        f12, nullf12, p12, f21, nullf21, p21 = compute_granger_trial_shuffling(
            job['d1'], job['d2'], max_lag_bins, n_shuffles=N_PSEUDO
        )
        return job['r1'], job['r2'], job['obj'], f12, nullf12, p12, f21, nullf21, p21

    print(f"Running Granger on {len(job_list)} pairs...")
    results = Parallel(n_jobs=N_CORES)(delayed(run_pair_job)(job) for job in job_list)

    session_dfs = []
    for r1, r2, obj, f12, nullf12, p12, f21, nullf21, p21 in results:
        temp_df = pd.DataFrame({
            'region_pair': [f'{r1} → {r2}', f'{r2} → {r1}'],
            'region1': [r1, r2],
            'region2': [r2, r1],
            'object': f'object{obj}',
            'p_value': [p12, p21],
            'f_stat': [f12, f21],
            'null_f_stat': [nullf12, nullf21]
        })
        temp_df['subject'] = subject
        temp_df['date'] = date
        session_dfs.append(temp_df)
    
    if session_dfs:
        granger_df = pd.concat([granger_df] + session_dfs, ignore_index=True)
        
    # Save to disk
    granger_df.to_csv(join(path_dict['save_path'], 'granger_causality_context.csv'), index=False)
    