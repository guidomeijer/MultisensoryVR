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
N_CORES = 12
N_SHUFFLES = 1000
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


def get_ar_matrices(target_trials, source_trials, lags):
    """
    Builds autoregression matrices (Y, X_res, X_source_lags) trial-by-trial.
    """
    n_trials, n_timebins = target_trials.shape
    if n_timebins <= (lags * 2 + 1):
        return None, None, None

    y_target_list = []
    X_res_list = []
    X_source_lags_list = []

    for trial in range(n_trials):
        target = target_trials[trial, :]
        source = source_trials[trial, :]
        y_lags = np.column_stack([target[lags - 1 - i: n_timebins - 1 - i] for i in range(lags)])
        x_lags = np.column_stack([source[lags - 1 - i: n_timebins - 1 - i] for i in range(lags)])
        
        y_target_list.append(target[lags:])
        X_res_list.append(np.column_stack([np.ones(n_timebins - lags), y_lags]))
        X_source_lags_list.append(x_lags)
        
    return y_target_list, X_res_list, X_source_lags_list


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
    Processes a single recording to extract trial-wise AR matrices.
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

    # Loop over region pairs to get AR data for this session
    session_ar_data = []
    for region1, region2 in combinations(residuals['object1'].keys(), 2):
        for this_obj in [1, 2, 3]:
            obj_str = f'object{this_obj}'
            res_data = residuals[obj_str]
            # r1 -> r2
            y_r2, x_res_r2, x_src_r2 = get_ar_matrices(res_data[region2], res_data[region1], max_lag_bins)
            if y_r2 is not None:
                session_ar_data.append({
                    'key': (region1, region2, obj_str),
                    'Y': y_r2, 'X_res': x_res_r2, 'X_src': x_src_r2
                })
            # r2 -> r1
            y_r1, x_res_r1, x_src_r1 = get_ar_matrices(res_data[region1], res_data[region2], max_lag_bins)
            if y_r1 is not None:
                session_ar_data.append({
                    'key': (region2, region1, obj_str),
                    'Y': y_r1, 'X_res': x_res_r1, 'X_src': x_src_r1
                })
    
    return session_ar_data


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

# Global storage for AR data across all sessions
global_pooled_data = {} # Key: (region1, region2, object_str), Value: [session_dict1, session_dict2, ...]

# Aggregate results from all recordings into the global storage
for recording_ar_data in all_recording_pooled_results:
    for res in recording_ar_data:
        key = res['key']
        if key not in global_pooled_data:
            global_pooled_data[key] = []
        global_pooled_data[key].append(res)


def calculate_f_stat(Y, X_res, X_full, lags): # Modified to return f_stat and var_explained_log_ratio
    f_stat = np.nan
    var_explained_log_ratio = np.nan
    try:
        _, rss_res, _, _ = np.linalg.lstsq(X_res, Y, rcond=None)
        _, rss_full, _, _ = np.linalg.lstsq(X_full, Y, rcond=None)
        
        if len(rss_res) > 0 and len(rss_full) > 0:
            rss_res_val = rss_res[0]
            rss_full_val = rss_full[0]

            # F-statistic calculation
            df_num = lags
            df_denom = len(Y) - X_full.shape[1]
            
            if df_denom > 0 and rss_full_val > 0: # Ensure valid denominator for F-stat
                if rss_res_val >= rss_full_val: # RSS_res should be >= RSS_full
                    f_stat = ((rss_res_val - rss_full_val) / df_num) / (rss_full_val / df_denom)
                    f_stat = max(0, f_stat)
                else: # This can happen due to numerical precision, treat as no improvement
                    f_stat = 0.0
            
            # Variance explained as log-ratio of RSS (log-ratio of residual variances)
            if rss_res_val > 0 and rss_full_val > 0:
                var_explained_log_ratio = np.log(rss_res_val / rss_full_val)
            
    except np.linalg.LinAlgError:
        pass
    except Exception as e:
        print(f"Error in calculate_f_stat: {e}")
        pass
        
    return f_stat, var_explained_log_ratio

# %% Calculate global Granger F-statistics after pooling all data
print(f"\nCalculating global Granger F-statistics and null distributions (N={N_SHUFFLES})...")
final_granger_results = []

for (r1, r2, obj_str), sessions in global_pooled_data.items():
    # Concatenate all pooled matrices for this specific (r1, r2, obj_str) across all sessions
    Y_all = np.concatenate([np.concatenate(s['Y']) for s in sessions])
    X_res_all = np.vstack([np.vstack(s['X_res']) for s in sessions])
    X_src_all = np.vstack([np.vstack(s['X_src']) for s in sessions])
    X_full_all = np.column_stack([X_res_all, X_src_all])

    real_f, real_var_explained = calculate_f_stat(Y_all, X_res_all, X_full_all, max_lag_bins)
    
    # Calculate null distribution by shuffling source trials within sessions
    null_f = []
    null_var_explained = []
    for _ in range(N_SHUFFLES):
        X_src_shuff_list = []
        for s in sessions:
            idx = np.random.permutation(len(s['X_src']))
            X_src_shuff_list.append(np.vstack([s['X_src'][i] for i in idx]))
        X_src_shuff = np.vstack(X_src_shuff_list)
        X_full_shuff = np.column_stack([X_res_all, X_src_shuff])
        shuff_f, shuff_var_explained = calculate_f_stat(Y_all, X_res_all, X_full_shuff, max_lag_bins)
        null_f.append(shuff_f)
        null_var_explained.append(shuff_var_explained)
    
    null_f = np.array(null_f)
    null_var_explained = np.array(null_var_explained)
    p_val_shuff = np.mean(null_f >= real_f) if not np.isnan(real_f) else np.nan

    final_granger_results.append({
        'region_pair': f'{r1} → {r2}',
        'region1': r1, 'region2': r2,
        'object': obj_str,
        'f_stat': real_f, # Real F-statistic
        'p_value_shuffled': p_val_shuff, # P-value from shuffled F-statistics
        'f_stat_null_95': np.nanpercentile(null_f, 95) if len(null_f) > 0 else np.nan, # 95th percentile of null F-stats
        'var_explained_log_ratio': real_var_explained, # Real variance explained (log-ratio)
        'var_explained_log_ratio_null_95': np.nanpercentile(null_var_explained, 95) if len(null_var_explained) > 0 else np.nan # 95th percentile of null variance explained
    })

granger_df = pd.DataFrame(final_granger_results)
granger_df.to_csv(join(path_dict['save_path'], 'granger_causality_context_global_pooled.csv'), index=False)
    