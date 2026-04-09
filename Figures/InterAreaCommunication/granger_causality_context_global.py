# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from itertools import combinations
from scipy import stats
from scipy.special import logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from msvr_functions import (paths, load_multiple_probes, load_subjects, load_objects,
                            calculate_peths)

# Settings
T_BEFORE = 2  # s
T_AFTER = 1
BIN_SIZE = 0.05
SMOOTHING = 0
MIN_NEURONS = 10
MIN_TRIALS = 30
N_CORES = 20
MAX_LAG = 0.5  # s
max_lag_bins = int(MAX_LAG / BIN_SIZE)

# Initialize
clf = RandomForestClassifier(random_state=42, n_jobs=1, n_estimators=500, max_depth=5)
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))


# Functions
def decode_stationary_context(this_x, this_y, this_clf, n_splits=10):
    """
    Decodes object identity by pooling all timebins together for training.

    Args:
        this_x (np.array): Data of shape (n_trials, n_neurons, n_timebins)
        this_y (np.array): Labels of shape (n_trials,)
        this_clf: Scikit-learn classifier instance.

    Returns:
        np.array: Decoding probabilities of shape (n_trials, n_timebins)
    """
    n_trials, n_neurons, n_timebins = this_x.shape
    decoding_probs = np.empty((n_trials, n_timebins))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(this_x, this_y):

        # Create training set by pooling together all time bins
        X_train_pooled = np.transpose(this_x[train_idx], (0, 2, 1)).reshape(-1, n_neurons)
        y_train_pooled = np.repeat(this_y[train_idx], n_timebins)

        # Train classifier
        this_clf.fit(X_train_pooled, y_train_pooled)

        # Predict on test trials
        X_test_pooled = np.transpose(this_x[test_idx], (0, 2, 1)).reshape(-1, n_neurons)
        probs_pooled = this_clf.predict_proba(X_test_pooled)
        probs_reshaped = probs_pooled.reshape(len(test_idx), n_timebins, -1)

        classes = this_clf.classes_
        true_class_indices = np.searchsorted(classes, this_y[test_idx])

        for i, true_idx in enumerate(true_class_indices):
            decoding_probs[test_idx[i], :] = probs_reshaped[i, :, true_idx]

    epsilon = 1e-5
    decoding_probs = np.clip(decoding_probs, epsilon, 1 - epsilon)
    return decoding_probs


def get_pooled_matrices(target_trials, source_trials, lags):
    """
    Builds pooled autoregression matrices (Y, X_restricted, X_full) across trials.

    Args:
        target_trials (np.array): Shape (n_trials, n_timebins)
        source_trials (np.array): Shape (n_trials, n_timebins)
        lags (int): Number of autoregressive lags

    Returns:
        Y_pooled, X_res_pooled, X_full_pooled: Pooled matrices for OLS.
    """
    n_trials, n_timebins = target_trials.shape
    if n_timebins <= (lags * 2 + 1):
        return None, None, None # Indicate invalid data

    y_target_pooled = []
    X_res_pooled = []
    X_full_pooled = []

    # Build design matrices per trial to prevent cross-trial bleeding
    for trial in range(n_trials):
        target = target_trials[trial, :]
        source = source_trials[trial, :]

        # Target vector for this trial
        y_target = target[lags:]

        # Lagged matrices
        y_lags = np.column_stack([target[lags - 1 - i: n_timebins - 1 - i] for i in range(lags)])
        x_lags = np.column_stack([source[lags - 1 - i: n_timebins - 1 - i] for i in range(lags)])

        # Restricted model: target ~ constant + target_lags
        X_res = np.column_stack([np.ones(n_timebins - lags), y_lags])
        # Full model: target ~ constant + target_lags + source_lags
        X_full = np.column_stack([X_res, x_lags])

        y_target_pooled.append(y_target)
        X_res_pooled.append(X_res)
        X_full_pooled.append(X_full)

    # Vertically stack all trials into massive pooled arrays
    Y_pooled = np.concatenate(y_target_pooled, axis=0)
    X_res_pooled = np.vstack(X_res_pooled)
    X_full_pooled = np.vstack(X_full_pooled)
    
    return Y_pooled, X_res_pooled, X_full_pooled


def get_session_pooled_matrices_for_pair(region1, region2, this_obj, residuals_obj_data, max_lag_bins):
    """
    Helper function to get pooled matrices for a single region pair and object within a session.
    """
    obj_str = f'object{this_obj}'
    
    results = []
    
    # Source -> Target (r1 -> r2)
    Y_r2, X_res_r2, X_full_r2 = get_pooled_matrices(
        residuals_obj_data[region2], residuals_obj_data[region1], max_lag_bins
    )
    if Y_r2 is not None:
        results.append({
            'key': (region1, region2, obj_str),
            'Y': Y_r2, 'X_res': X_res_r2, 'X_full': X_full_r2
        })

    # Target -> Source (r2 -> r1)
    Y_r1, X_res_r1, X_full_r1 = get_pooled_matrices(
        residuals_obj_data[region1], residuals_obj_data[region2], max_lag_bins
    )
    if Y_r1 is not None:
        results.append({
            'key': (region2, region1, obj_str),
            'Y': Y_r1, 'X_res': X_res_r1, 'X_full': X_full_r1
        })
        
    return results


def process_single_region(region, probe, spikes, clusters, all_obj_df, clf):
    """
    Helper function to process decoding for a single region.
    """
    region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
    if region_neurons.shape[0] < MIN_NEURONS:
        return region, None

    region_results = {}
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
        
        # Decode context per timebin
        X = np.concatenate([soundA, soundB], axis=0)
        y = np.concatenate([np.zeros(soundA.shape[0]), np.ones(soundB.shape[0])]).astype(int)
        
        if X.shape[0] < MIN_TRIALS:
            region_results[f'object{this_obj}'] = None
            continue

        trial_probs = decode_stationary_context(X, y, clf)
        trial_probs = logit(trial_probs)

        # Subtract class-specific mean
        subtracted_probs = np.zeros_like(trial_probs)
        for label in [0, 1]:
            mask = (y == label)
            subtracted_probs[mask, :] = trial_probs[mask, :] - np.mean(trial_probs[mask, :], axis=0)
        
        region_results[f'object{this_obj}'] = subtracted_probs
    return region, region_results


def process_single_recording(subject, date, path_dict, clf, max_lag_bins, MIN_NEURONS, MIN_TRIALS):
    """
    Processes a single recording to extract pooled autoregression matrices.
    """
    print(f'\nProcessing {subject} {date}...')

    # Load in data
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_multiple_probes(session_path)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)
    
    if trials.shape[0] < MIN_TRIALS:
        return [] # Return empty list if not enough trials
    
    # %% Loop over regions
    
    # Get list of all regions and which probe they were recorded on
    regions, region_probes = [], []
    for p, probe in enumerate(spikes.keys()):
        regions.append(np.unique(clusters[probe]['region']))
        region_probes.append([probe] * np.unique(clusters[probe]['region']).shape[0])
    regions = np.concatenate(regions)
    region_probes = np.concatenate(region_probes)
    
    # Process regions sequentially within this recording (since recordings are parallelized externally)
    decoded_outputs = []
    for reg, prob in zip(regions, region_probes):
        if reg != 'root':
            decoded_outputs.append(process_single_region(reg, prob, spikes, clusters, all_obj_df, clf))

    # Reconstruct residuals dictionary
    residuals = {f'object{i}': {} for i in [1, 2, 3]}
    for region, results in decoded_outputs:
        if results is None: continue
        for this_obj in [1, 2, 3]:
            if results[f'object{this_obj}'] is not None:
                residuals[f'object{this_obj}'][region] = results[f'object{this_obj}']

    # Loop over region pairs to get pooled matrices for this session
    session_pair_jobs = []
    for region1, region2 in combinations(residuals['object1'].keys(), 2):
        for this_obj in [1, 2, 3]:
            session_pair_jobs.append({
                'r1': region1, 'r2': region2,
                'obj': this_obj,
                'residuals_obj_data': residuals[f'object{this_obj}']
            })

    session_pooled_results_for_this_recording = []
    for job in session_pair_jobs:
        session_pooled_results_for_this_recording.extend(
            get_session_pooled_matrices_for_pair(
                job['r1'], job['r2'], job['obj'], job['residuals_obj_data'], max_lag_bins
            )
        )
    
    return session_pooled_results_for_this_recording


# %% Parallelize over recordings

# Prepare arguments for parallel processing of recordings
recording_jobs_args = []
for subject, date in zip(rec['subject'], rec['date']):
    recording_jobs_args.append({
        'subject': subject,
        'date': date,
        'path_dict': path_dict,
        'clf': clf,
        'max_lag_bins': max_lag_bins,
        'MIN_NEURONS': MIN_NEURONS,
        'MIN_TRIALS': MIN_TRIALS
    })

print(f"Processing {len(recording_jobs_args)} recordings in parallel...")
all_recording_pooled_results = Parallel(n_jobs=N_CORES)(
    delayed(process_single_recording)(**job_args) for job_args in recording_jobs_args
)

# Global storage for pooled matrices across all sessions
global_pooled_data = {} # Key: (region1, region2, object_str), Value: {'Y': [], 'X_res': [], 'X_full': []}

# Aggregate results from all recordings into the global storage
for recording_results_list in all_recording_pooled_results:
    for res in recording_results_list:
        key = res['key']
        if key not in global_pooled_data:
            global_pooled_data[key] = {'Y': [], 'X_res': [], 'X_full': []}
        global_pooled_data[key]['Y'].append(res['Y'])
        global_pooled_data[key]['X_res'].append(res['X_res'])
        global_pooled_data[key]['X_full'].append(res['X_full'])


# %% Calculate global Granger F-statistics after pooling all data
print("\nCalculating global Granger F-statistics...")
final_granger_results = []

for (r1, r2, obj_str), data_lists in global_pooled_data.items():
    # Concatenate all pooled matrices for this specific (r1, r2, obj_str) across all sessions
    Y_global = np.concatenate(data_lists['Y'], axis=0)
    X_res_global = np.vstack(data_lists['X_res'])
    X_full_global = np.vstack(data_lists['X_full'])

    f_stat, p_value = np.nan, np.nan
    try:
        # Solve OLS on the globally pooled data
        _, rss_res, _, _ = np.linalg.lstsq(X_res_global, Y_global, rcond=None)
        _, rss_full, _, _ = np.linalg.lstsq(X_full_global, Y_global, rcond=None)

        if len(rss_res) > 0 and len(rss_full) > 0:
            # Pooled Degrees of Freedom
            n_total_observations = len(Y_global)
            df_num = max_lag_bins
            df_denom = n_total_observations - (1 + 2 * max_lag_bins)

            if df_denom > 0: # Ensure denominator is positive for F-stat calculation
                f_stat = ((rss_res[0] - rss_full[0]) / df_num) / (rss_full[0] / df_denom)
                f_stat = max(0, f_stat) # F-stat should not be negative
                p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)

    except np.linalg.LinAlgError:
        pass # f_stat and p_value remain nan

    final_granger_results.append({
        'region_pair': f'{r1} → {r2}',
        'region1': r1, 'region2': r2,
        'object': obj_str,
        'p_value': p_value, 'f_stat': f_stat
    })

granger_df = pd.DataFrame(final_granger_results)
granger_df.to_csv(join(path_dict['save_path'], 'granger_causality_context_global_pooled.csv'), index=False)
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
    
    # Parallelize decoding across regions
    print(f'Decoding {len(regions)} regions in parallel...')
    decoded_outputs = Parallel(n_jobs=N_CORES)(
        delayed(process_single_region)(reg, prob, spikes, clusters, all_obj_df, clf) 
        for reg, prob in zip(regions, region_probes) if reg != 'root'
    )

    # Reconstruct residuals dictionary
    residuals = {f'object{i}': {} for i in [1, 2, 3]}
    for region, results in decoded_outputs:
        if results is None: continue
        for this_obj in [1, 2, 3]:
            if results[f'object{this_obj}'] is not None:
                residuals[f'object{this_obj}'][region] = results[f'object{this_obj}']

    # Loop over region pairs
    session_pair_jobs = []
    for region1, region2 in combinations(residuals['object1'].keys(), 2):
        for this_obj in [1, 2, 3]:
            session_pair_jobs.append({
                'r1': region1, 'r2': region2,
                'obj': this_obj,
                'residuals_obj_data': residuals[f'object{this_obj}']
            })

    print(f"Collecting pooled matrices for {len(session_pair_jobs)} pairs in session {subject} {date}...")
    session_pooled_results = Parallel(n_jobs=N_CORES)(
        delayed(get_session_pooled_matrices_for_pair)(
            job['r1'], job['r2'], job['obj'], job['residuals_obj_data'], max_lag_bins
        ) for job in session_pair_jobs
    )

    # Aggregate results from this session into the global storage
    for session_results_list in session_pooled_results:
        for res in session_results_list:
            key = res['key']
            if key not in global_pooled_data:
                global_pooled_data[key] = {'Y': [], 'X_res': [], 'X_full': []}
            global_pooled_data[key]['Y'].append(res['Y'])
            global_pooled_data[key]['X_res'].append(res['X_res'])
            global_pooled_data[key]['X_full'].append(res['X_full'])


# %% Calculate global Granger F-statistics after pooling all data
print("\nCalculating global Granger F-statistics...")
final_granger_results = []

for (r1, r2, obj_str), data_lists in global_pooled_data.items():
    # Concatenate all pooled matrices for this specific (r1, r2, obj_str) across all sessions
    Y_global = np.concatenate(data_lists['Y'], axis=0)
    X_res_global = np.vstack(data_lists['X_res'])
    X_full_global = np.vstack(data_lists['X_full'])

    f_stat, p_value = np.nan, np.nan
    try:
        # Solve OLS on the globally pooled data
        _, rss_res, _, _ = np.linalg.lstsq(X_res_global, Y_global, rcond=None)
        _, rss_full, _, _ = np.linalg.lstsq(X_full_global, Y_global, rcond=None)

        if len(rss_res) > 0 and len(rss_full) > 0:
            # Pooled Degrees of Freedom
            n_total_observations = len(Y_global)
            df_num = max_lag_bins
            df_denom = n_total_observations - (1 + 2 * max_lag_bins)

            if df_denom > 0: # Ensure denominator is positive for F-stat calculation
                f_stat = ((rss_res[0] - rss_full[0]) / df_num) / (rss_full[0] / df_denom)
                f_stat = max(0, f_stat) # F-stat should not be negative
                p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)

    except np.linalg.LinAlgError:
        pass # f_stat and p_value remain nan

    final_granger_results.append({
        'region_pair': f'{r1} → {r2}',
        'region1': r1, 'region2': r2,
        'object': obj_str,
        'p_value': p_value, 'f_stat': f_stat
    })

granger_df = pd.DataFrame(final_granger_results)
granger_df.to_csv(join(path_dict['save_path'], 'granger_causality_context_global_pooled.csv'), index=False)
    