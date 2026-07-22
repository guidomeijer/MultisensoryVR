# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 2026
Author: Guido Meijer

Reduced Rank Regression (RRR) analysis of directional inter-area communication.
Fits RRR on a sliding window of 5 timebins grouped together for each object separately.
Calculates cross-validated R-squared and optimal dimensionality over time for all directional region pairs.
"""

import numpy as np
import pandas as pd
from itertools import permutations
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
TIME_WIN = {'obj1': [-2, 2], 'obj2': [-2, 2], 'obj3': [-2, 2]}
BIN_SIZE = 0.025  # 25 ms
SMOOTHING = 0.05  # 50 ms
WINDOW_SIZE = 5   # 5 timebins (125 ms)
MIN_NEURONS = 5   # per region
MAX_DIM = 10
N_CV_FOLDS = 5
N_CPUS = -6

# Initialize
path_dict = paths()
subjects = load_subjects()
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv', dtype={'subject': str, 'date': str})
rec = rec.drop_duplicates(subset=['subject', 'date'])

def fit_rrr_window(X_win, Y_win, max_dim=MAX_DIM, n_splits=N_CV_FOLDS):
    """
    Fits Reduced Rank Regression on X_win -> Y_win using KFold cross-validation over trials.
    X_win shape: (n_trials, window_size, n_source_neurons)
    Y_win shape: (n_trials, window_size, n_target_neurons)
    Returns:
        max_r2: Peak cross-validated R^2 across ranks
        d_opt: Optimal rank (dimensionality) associated with peak R^2
    """
    n_trials, win_size, P = X_win.shape
    _, _, Q = Y_win.shape
    
    if n_trials < n_splits or P < 1 or Q < 1:
        return np.nan, np.nan
        
    use_max_dim = min(P, Q, max_dim)
    if use_max_dim < 1:
        return np.nan, np.nan

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    Y_pred_cv = np.zeros((use_max_dim, n_trials * win_size, Q))
    Y_true_cv = np.zeros((n_trials * win_size, Q))
    
    ptr = 0
    for train_idx, test_idx in kf.split(np.arange(n_trials)):
        n_test = len(test_idx)
        n_test_samples = n_test * win_size
        
        X_train = X_win[train_idx].reshape(-1, P)
        Y_train = Y_win[train_idx].reshape(-1, Q)
        X_test = X_win[test_idx].reshape(-1, P)
        Y_test = Y_win[test_idx].reshape(-1, Q)
        
        Y_true_cv[ptr : ptr + n_test_samples] = Y_test
        
        # OLS estimation
        try:
            B_ols = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]
        except np.linalg.LinAlgError:
            B_ols = np.zeros((P, Q))
            
        Y_train_hat = X_train @ B_ols
        
        # SVD on predicted training targets
        try:
            _, _, Vt = np.linalg.svd(Y_train_hat, full_matrices=False)
            V = Vt.T
        except np.linalg.LinAlgError:
            V = np.eye(Q, use_max_dim)
            
        for d in range(1, use_max_dim + 1):
            V_d = V[:, :d]
            B_d = B_ols @ V_d @ V_d.T
            Y_test_hat_d = X_test @ B_d
            Y_pred_cv[d - 1, ptr : ptr + n_test_samples, :] = Y_test_hat_d
            
        ptr += n_test_samples
        
    # Evaluate per-neuron R^2 over timebins and average across target neurons
    r2_cv = np.zeros(use_max_dim)
    y_mean = np.mean(Y_true_cv, axis=0)
    ss_tot = np.sum((Y_true_cv - y_mean)**2, axis=0)
    
    for d in range(use_max_dim):
        ss_res = np.sum((Y_true_cv - Y_pred_cv[d])**2, axis=0)
        # Avoid division by zero for silent target channels
        valid_neurons = ss_tot > 1e-10
        if np.any(valid_neurons):
            r2_per_neuron = 1 - (ss_res[valid_neurons] / ss_tot[valid_neurons])
            r2_cv[d] = np.mean(r2_per_neuron)
        else:
            r2_cv[d] = np.nan
            
    if np.all(np.isnan(r2_cv)):
        return np.nan, np.nan
        
    d_opt = int(np.nanargmax(r2_cv) + 1)
    max_r2 = float(r2_cv[d_opt - 1])
    
    return max_r2, d_opt

def process_session(subject, date):
    print(f'Processing session {subject} {date}...')
    session_path = path_dict['local_data_path'] / 'Subjects' / str(subject) / str(date)
    spikes, clusters, channels = load_multiple_probes(session_path)
    all_obj_df = load_objects(subject, date)

    spikes_dict = {'obj1': dict(), 'obj2': dict(), 'obj3': dict()}
    
    for k, probe in enumerate(spikes.keys()):
        for j, region in enumerate(np.unique(clusters[probe]['region'])):
            if region == 'root':
                continue
            
            region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]

            binned_spikes_tmp = {}
            active_mask = np.zeros(len(region_neurons), dtype=bool)
            
            for m, obj in enumerate(['obj1', 'obj2', 'obj3']):
                peth, binned_spikes = calculate_peths(
                    spikes[probe]['times'], spikes[probe]['clusters'], region_neurons,
                    all_obj_df.loc[all_obj_df['object'] == m+1, 'times'].values,
                    np.abs(TIME_WIN[obj][0]), TIME_WIN[obj][1], BIN_SIZE, SMOOTHING, return_fr=False)
                
                binned_spikes_tmp[obj] = np.swapaxes(binned_spikes, 1, 2)
                active_mask = active_mask | (np.max(peth['means'], axis=1) > 0.01)
                
            if np.sum(active_mask) < MIN_NEURONS:
                continue
                
            for obj in ['obj1', 'obj2', 'obj3']:
                spikes_dropped = binned_spikes_tmp[obj][:, :, active_mask]
                spikes_dict[obj][region] = spikes_dropped

    # Determine available regions common across objects
    regions_set = set(spikes_dict['obj1'].keys()).intersection(
        spikes_dict['obj2'].keys()
    ).intersection(
        spikes_dict['obj3'].keys()
    )
    
    available_regions = sorted(list(regions_set))
    if len(available_regions) < 2:
        return pd.DataFrame()

    # Generate all directional pairs (A -> B)
    directional_pairs = list(permutations(available_regions, 2))
    
    n_timebins = spikes_dict['obj1'][available_regions[0]].shape[1]
    time_ax = np.arange(n_timebins) * BIN_SIZE + TIME_WIN['obj1'][0] + (BIN_SIZE / 2)
    n_windows = n_timebins - WINDOW_SIZE + 1
    window_time_centers = time_ax[WINDOW_SIZE // 2 : WINDOW_SIZE // 2 + n_windows]
    
    rows = []
    
    for obj in ['obj1', 'obj2', 'obj3']:
        for src_region, tgt_region in directional_pairs:
            X_data = spikes_dict[obj][src_region]  # (n_trials, n_timebins, P)
            Y_data = spikes_dict[obj][tgt_region]  # (n_trials, n_timebins, Q)
            
            for w_idx in range(n_windows):
                w_start = w_idx
                w_end = w_idx + WINDOW_SIZE
                
                X_win = X_data[:, w_start:w_end, :]
                Y_win = Y_data[:, w_start:w_end, :]
                
                max_r2, d_opt = fit_rrr_window(X_win, Y_win, max_dim=MAX_DIM, n_splits=N_CV_FOLDS)
                
                rows.append({
                    'subject': subject,
                    'date': date,
                    'object': obj,
                    'source_region': src_region,
                    'target_region': tgt_region,
                    'region_pair': f'{src_region} → {tgt_region}',
                    'time': window_time_centers[w_idx],
                    'r2': max_r2,
                    'dimensionality': d_opt
                })
                
    return pd.DataFrame(rows)

if __name__ == '__main__':
    print(f'Processing {len(rec)} sessions sequentially...')
    
    results = Parallel(n_jobs=N_CPUS)(
        delayed(process_session)(row['subject'], row['date'])
        for _, row in rec.iterrows()
    )
    
    results = [df for df in results if not df.empty]
    
    if results:
        rrr_df = pd.concat(results, ignore_index=True)
        out_path = path_dict['google_drive_data_path'] / 'rrr_results.csv'
        rrr_df.to_csv(out_path, index=False)
        print(f'Processing complete. Saved {len(rrr_df)} rows to {out_path}')
    else:
        print('No sessions had sufficient regions for RRR analysis.')
