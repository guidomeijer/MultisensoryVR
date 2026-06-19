# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:30:00 2026 by Guido Meijer
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSCanonical
from sklearn.model_selection import LeaveOneGroupOut, KFold, cross_val_predict
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects, bin_signal

# Settings
TIME_WIN = {'obj1': [-2, 2], 'obj2': [-2, 2], 'obj3': [-2, 2]}
BIN_SIZE = 0.025
SMOOTHING = 0.05
MIN_NEURONS = 5  # per region
N_CPUS = -6
N_COMPONENTS = 4
DECODING_BIN_SIZE = 0.3
DECODING_BIN_SHIFT = 0.05
CORTICAL_REGIONS = ['VIS', 'PERI', 'TEa', 'AUD', 'LEC']
TARGET_REGION = 'CA1'

# Session selection
SUBJECT = '478154'
DATE = '20251003'

# Initialize
path_dict = paths()
subjects = load_subjects()
clf = RandomForestClassifier(random_state=42, n_jobs=1, n_estimators=20, max_depth=5)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Load in data
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv', dtype={'subject': str, 'date': str})
rec = rec.drop_duplicates(subset=['subject', 'date'])

# Contruct bin centers for decoding
# Here we will use a common "time since object onset" axis from 0 to 2 seconds
TIME_WIN_COMMON = [-2, 2]
bin_centers_common = np.arange(TIME_WIN_COMMON[0] + (DECODING_BIN_SIZE / 2),
                               TIME_WIN_COMMON[1] - (DECODING_BIN_SIZE / 2) + DECODING_BIN_SHIFT,
                               step=DECODING_BIN_SHIFT)

def _fit_single_trial(x_region_a, x_region_b, train_idx, test_idx, use_n_components, n_neurons_X, n_neurons_Y):
    X_train_2D = x_region_a[train_idx].reshape(-1, n_neurons_X)
    Y_train_2D = x_region_b[train_idx].reshape(-1, n_neurons_Y)
    X_test_2D = x_region_a[test_idx].reshape(-1, n_neurons_X)
    Y_test_2D = x_region_b[test_idx].reshape(-1, n_neurons_Y)

    pls = PLSCanonical(n_components=use_n_components)
    pls.fit(X_train_2D, Y_train_2D)
    X_latent_test, Y_latent_test = pls.transform(X_test_2D, Y_test_2D)
    
    return test_idx[0], X_latent_test, Y_latent_test

def fit_pla(x_region_a, x_region_b, use_n_components):
    """
    Calculate the inter-regional coupling strength using PLS-Canonical and leave-one-trial-out cross-validation.
    """
    n_trials, n_timebins, n_neurons_X = x_region_a.shape
    _, _, n_neurons_Y = x_region_b.shape

    X_latents = np.zeros((n_trials, n_timebins, use_n_components))
    Y_latents = np.zeros((n_trials, n_timebins, use_n_components))

    logo = LeaveOneGroupOut()

    results = Parallel(n_jobs=N_CPUS)(
        delayed(_fit_single_trial)(
            x_region_a, x_region_b, train_idx, test_idx, use_n_components, n_neurons_X, n_neurons_Y
        )
        for train_idx, test_idx in logo.split(x_region_a, x_region_b, groups=np.arange(n_trials))
    )

    for current_trial, X_latent_test, Y_latent_test in results:
        X_latents[current_trial] = X_latent_test
        Y_latents[current_trial] = Y_latent_test

    return X_latents, Y_latents

def _decode_timebin(tt, decode_cortex_tt, decode_ca1_tt, trial_sounds, trial_objects, cv, clf):
    # Select trials for object 1
    mask_obj1 = (trial_objects == 'obj1')
    decode_cortex_tt_obj1 = decode_cortex_tt[mask_obj1]
    decode_ca1_tt_obj1 = decode_ca1_tt[mask_obj1]
    trial_sounds_obj1 = trial_sounds[mask_obj1]

    # Select trials for object 2
    mask_obj2 = (trial_objects == 'obj2')
    decode_cortex_tt_obj2 = decode_cortex_tt[mask_obj2]
    decode_ca1_tt_obj2 = decode_ca1_tt[mask_obj2]
    trial_sounds_obj2 = trial_sounds[mask_obj2]

    # Select trials for object 3
    mask_obj3 = (trial_objects == 'obj3')
    decode_cortex_tt_obj3 = decode_cortex_tt[mask_obj3]
    decode_ca1_tt_obj3 = decode_ca1_tt[mask_obj3]
    trial_sounds_obj3 = trial_sounds[mask_obj3]

    # Context (sound) decoding for object 1
    if len(trial_sounds_obj1) >= 5 and len(np.unique(trial_sounds_obj1)) > 1:
        pred_ctx_sound_obj1 = cross_val_predict(clf, decode_cortex_tt_obj1, trial_sounds_obj1, cv=cv, n_jobs=1)
        pred_ca1_sound_obj1 = cross_val_predict(clf, decode_ca1_tt_obj1, trial_sounds_obj1, cv=cv, n_jobs=1)
        acc_ctx_sound_obj1 = accuracy_score(trial_sounds_obj1, pred_ctx_sound_obj1)
        acc_ca1_sound_obj1 = accuracy_score(trial_sounds_obj1, pred_ca1_sound_obj1)
    else:
        acc_ctx_sound_obj1 = np.nan
        acc_ca1_sound_obj1 = np.nan

    # Context (sound) decoding for object 2
    if len(trial_sounds_obj2) >= 5 and len(np.unique(trial_sounds_obj2)) > 1:
        pred_ctx_sound_obj2 = cross_val_predict(clf, decode_cortex_tt_obj2, trial_sounds_obj2, cv=cv, n_jobs=1)
        pred_ca1_sound_obj2 = cross_val_predict(clf, decode_ca1_tt_obj2, trial_sounds_obj2, cv=cv, n_jobs=1)
        acc_ctx_sound_obj2 = accuracy_score(trial_sounds_obj2, pred_ctx_sound_obj2)
        acc_ca1_sound_obj2 = accuracy_score(trial_sounds_obj2, pred_ca1_sound_obj2)
    else:
        acc_ctx_sound_obj2 = np.nan
        acc_ca1_sound_obj2 = np.nan

    # Context (sound) decoding for object 3
    if len(trial_sounds_obj3) >= 5 and len(np.unique(trial_sounds_obj3)) > 1:
        pred_ctx_sound_obj3 = cross_val_predict(clf, decode_cortex_tt_obj3, trial_sounds_obj3, cv=cv, n_jobs=1)
        pred_ca1_sound_obj3 = cross_val_predict(clf, decode_ca1_tt_obj3, trial_sounds_obj3, cv=cv, n_jobs=1)
        acc_ctx_sound_obj3 = accuracy_score(trial_sounds_obj3, pred_ctx_sound_obj3)
        acc_ca1_sound_obj3 = accuracy_score(trial_sounds_obj3, pred_ca1_sound_obj3)
    else:
        acc_ctx_sound_obj3 = np.nan
        acc_ca1_sound_obj3 = np.nan

    # Object decoding
    pred_ctx_obj = cross_val_predict(clf, decode_cortex_tt, trial_objects, cv=cv, n_jobs=1)
    pred_ca1_obj = cross_val_predict(clf, decode_ca1_tt, trial_objects, cv=cv, n_jobs=1)
    acc_ctx_obj = accuracy_score(trial_objects, pred_ctx_obj)
    acc_ca1_obj = accuracy_score(trial_objects, pred_ca1_obj)

    return (tt, acc_ctx_sound_obj1, acc_ca1_sound_obj1,
            acc_ctx_sound_obj2, acc_ca1_sound_obj2,
            acc_ctx_sound_obj3, acc_ca1_sound_obj3,
            acc_ctx_obj, acc_ca1_obj)

# %% Main script

session_path = path_dict['local_data_path'] / 'Subjects' / str(SUBJECT) / str(DATE)
spikes, clusters, channels = load_multiple_probes(session_path)
all_obj_df = load_objects(SUBJECT, DATE)

spikes_dict = {'obj1': dict(), 'obj2': dict(), 'obj3': dict()}

for k, probe in enumerate(spikes.keys()):
    for j, region in enumerate(np.unique(clusters[probe]['region'])):
        if region == 'root':
            continue
        
        region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]

        # First, calculate PETHs for both objects to determine active neurons
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

for ctx_region in CORTICAL_REGIONS:
    if ctx_region not in spikes_dict['obj1'] or ctx_region not in spikes_dict['obj2'] or ctx_region not in spikes_dict['obj3']:
        continue
        
    # Combine obj1, obj2, and obj3 data to fit a common PLA subspace
    x_cortex = np.concatenate((spikes_dict['obj1'][ctx_region], spikes_dict['obj2'][ctx_region], spikes_dict['obj3'][ctx_region]), axis=0)
    y_ca1 = np.concatenate((spikes_dict['obj1']['CA1'], spikes_dict['obj2']['CA1'], spikes_dict['obj3']['CA1']), axis=0)
    
    # Get latents
    latents_cortex, latents_ca1 = fit_pla(x_cortex, y_ca1, use_n_components=N_COMPONENTS)
    
    # Labels for decoding
    trial_sounds = np.concatenate((
        all_obj_df.loc[all_obj_df['object'] == 1, 'sound'].values,
        all_obj_df.loc[all_obj_df['object'] == 2, 'sound'].values,
        all_obj_df.loc[all_obj_df['object'] == 3, 'sound'].values
    ))
    
    trial_objects = np.concatenate((
        np.full(spikes_dict['obj1'][ctx_region].shape[0], 'obj1'),
        np.full(spikes_dict['obj2'][ctx_region].shape[0], 'obj2'),
        np.full(spikes_dict['obj3'][ctx_region].shape[0], 'obj3')
    ))
    
    # We need to bin latents for decoding
    time_ax_common = np.arange(x_cortex.shape[1]) * BIN_SIZE + TIME_WIN_COMMON[0] + (BIN_SIZE / 2)
    
    decode_cortex = np.full((latents_cortex.shape[0], bin_centers_common.shape[0], latents_cortex.shape[2]), np.nan)
    decode_ca1 = np.full((latents_ca1.shape[0], bin_centers_common.shape[0], latents_ca1.shape[2]), np.nan)
    
    for kk in range(latents_cortex.shape[2]):
        for tt in range(latents_cortex.shape[0]):
            decode_cortex[tt, :, kk] = bin_signal(time_ax_common, latents_cortex[tt, :, kk], bin_centers_common, DECODING_BIN_SIZE)
            decode_ca1[tt, :, kk] = bin_signal(time_ax_common, latents_ca1[tt, :, kk], bin_centers_common, DECODING_BIN_SIZE)

    # Reshape into (trials x latents x timebins)
    decode_cortex = np.swapaxes(decode_cortex, 1, 2)
    decode_ca1 = np.swapaxes(decode_ca1, 1, 2)
    
    accuracy_cortex_sound_obj1 = np.full(decode_cortex.shape[2], np.nan)
    accuracy_ca1_sound_obj1 = np.full(decode_ca1.shape[2], np.nan)
    accuracy_cortex_sound_obj2 = np.full(decode_cortex.shape[2], np.nan)
    accuracy_ca1_sound_obj2 = np.full(decode_ca1.shape[2], np.nan)
    accuracy_cortex_sound_obj3 = np.full(decode_cortex.shape[2], np.nan)
    accuracy_ca1_sound_obj3 = np.full(decode_ca1.shape[2], np.nan)
    accuracy_cortex_obj = np.full(decode_cortex.shape[2], np.nan)
    accuracy_ca1_obj = np.full(decode_ca1.shape[2], np.nan)

    decode_results = Parallel(n_jobs=N_CPUS)(
        delayed(_decode_timebin)(
            tt, decode_cortex[:, :, tt], decode_ca1[:, :, tt], trial_sounds, trial_objects, cv, clf
        )
        for tt in range(decode_cortex.shape[2])
    )
    
    for (tt, acc_c_s1, acc_ca_s1, acc_c_s2, acc_ca_s2, acc_c_s3, acc_ca_s3, acc_c_o, acc_ca_o) in decode_results:
        accuracy_cortex_sound_obj1[tt] = acc_c_s1
        accuracy_ca1_sound_obj1[tt] = acc_ca_s1
        accuracy_cortex_sound_obj2[tt] = acc_c_s2
        accuracy_ca1_sound_obj2[tt] = acc_ca_s2
        accuracy_cortex_sound_obj3[tt] = acc_c_s3
        accuracy_ca1_sound_obj3[tt] = acc_ca_s3
        accuracy_cortex_obj[tt] = acc_c_o
        accuracy_ca1_obj[tt] = acc_ca_o

    session_pla_df = pd.concat([session_pla_df, pd.DataFrame(data={
        'accuracy': np.concatenate((
            accuracy_cortex_sound_obj1, accuracy_ca1_sound_obj1,
            accuracy_cortex_sound_obj2, accuracy_ca1_sound_obj2,
            accuracy_cortex_sound_obj3, accuracy_ca1_sound_obj3,
            accuracy_cortex_obj, accuracy_ca1_obj
        )),
        'time': np.tile(bin_centers_common, 8),
        'region': np.concatenate((
            np.full(accuracy_cortex_sound_obj1.shape, ctx_region),
            np.full(accuracy_ca1_sound_obj1.shape, 'CA1'),
            np.full(accuracy_cortex_sound_obj2.shape, ctx_region),
            np.full(accuracy_ca1_sound_obj2.shape, 'CA1'),
            np.full(accuracy_cortex_sound_obj3.shape, ctx_region),
            np.full(accuracy_ca1_sound_obj3.shape, 'CA1'),
            np.full(accuracy_cortex_obj.shape, ctx_region),
            np.full(accuracy_ca1_obj.shape, 'CA1')
        )),
        'decoder': np.concatenate((
            np.full(accuracy_cortex_sound_obj1.shape[0] * 2, 'context_obj1'),
            np.full(accuracy_cortex_sound_obj2.shape[0] * 2, 'context_obj2'),
            np.full(accuracy_cortex_sound_obj3.shape[0] * 2, 'context_obj3'),
            np.full(accuracy_cortex_obj.shape[0] * 2, 'object')
        )),
        'cortical_region': ctx_region,
        'subject': subject,
        'date': date
    })])

