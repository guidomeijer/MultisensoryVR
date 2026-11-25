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


def pooled_bivariate_gc_f(target_trials, source_trials, lags):
    """
    Computes Granger Causality F-statistic by pooling autoregression across trials.

    Args:
        target_trials (np.array): Shape (n_trials, n_timebins)
        source_trials (np.array): Shape (n_trials, n_timebins)
        lags (int): Number of autoregressive lags

    Returns:
        f_stat: F-statistic for Source -> Target causality.
        p_value: Significance of the F-statistic.
    """
    n_trials, n_timebins = target_trials.shape

    if n_timebins <= (lags * 2 + 1):
        return np.nan, np.nan

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

    try:
        # Solve OLS on the pooled data
        _, rss_res, _, _ = np.linalg.lstsq(X_res_pooled, Y_pooled, rcond=None)
        _, rss_full, _, _ = np.linalg.lstsq(X_full_pooled, Y_pooled, rcond=None)

        if len(rss_res) == 0 or len(rss_full) == 0:
            return np.nan, np.nan

        # Pooled Degrees of Freedom
        n_total_observations = len(Y_pooled)
        df_num = lags
        df_denom = n_total_observations - (1 + 2 * lags)

        f_stat = ((rss_res[0] - rss_full[0]) / df_num) / (rss_full[0] / df_denom)
        f_stat = max(0, f_stat)
        p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)

        return f_stat, p_value

    except np.linalg.LinAlgError:
        return np.nan, np.nan

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
        f12, p12 = pooled_bivariate_gc_f(
            job['d2'], job['d1'], max_lag_bins
        )
        f21, p21 = pooled_bivariate_gc_f(
            job['d1'], job['d2'], max_lag_bins
        )
        return job['r1'], job['r2'], job['obj'], f12, p12, f21, p21

    print(f"Running Granger on {len(job_list)} pairs...")
    results = Parallel(n_jobs=N_CORES)(delayed(run_pair_job)(job) for job in job_list)

    session_dfs = []
    for r1, r2, obj, f12, p12, f21, p21 in results:
        temp_df = pd.DataFrame({
            'region_pair': [f'{r1} → {r2}', f'{r2} → {r1}'],
            'region1': [r1, r2],
            'region2': [r2, r1],
            'object': f'object{obj}',
            'p_value': [p12, p21],
            'f_stat': [f12, f21]
        })
        temp_df['subject'] = subject
        temp_df['date'] = date
        session_dfs.append(temp_df)
    
    if session_dfs:
        granger_df = pd.concat([granger_df] + session_dfs, ignore_index=True)
        
    # Save to disk
    granger_df.to_csv(join(path_dict['save_path'], 'granger_causality_context.csv'), index=False)
    