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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from msvr_functions import (paths, load_multiple_probes, load_subjects, load_objects,
                            get_spike_counts_in_bins)

# Settings
D_BEFORE = 300  # mm
D_AFTER = 300
BIN_SIZE = 50
STEP_SIZE = 7
MIN_NEURONS = 10
MIN_TRIALS = 30
MIN_SPEED = 20  # mm/s
N_CORES = 12
MAX_LAG = 100  # mm
N_PSEUDO = 500

# Create distance array
d_centers = np.arange(-D_BEFORE + (BIN_SIZE/2), D_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=1))
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))


# Functions for parallelization
def decode_context(X, y, clf):
    """
    Decodes object identity per spatial bin using efficient leave-one-out cross-validation.

    Args:
        X (np.array): Data of shape (n_trials, n_neurons, n_bins)
        y (np.array): Labels of shape (n_trials,)
        clf: Scikit-learn classifier instance.

    Returns:
        np.array: Decoding probabilities of shape (n_trials, n_bins)
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
    res = model.fit(maxlags=int(MAX_LAG / STEP_SIZE))

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


def compute_granger_trial_shuffling(residuals_r1, residuals_r2, max_lag_bins, n_shuffles=500):
    """
    Computes Granger Causality using trial shuffling on residuals.

    Args:
        residuals_r1 (np.array): Shape (n_trials, n_timepoints)
        residuals_r2 (np.array): Shape (n_trials, n_timepoints)
        max_lag_bins (int): Max lag for VAR model.
        n_shuffles (int): Number of surrogates.

    Returns:
        tuple: (real_f12, real_f21, p_12, p_21)
    """
    n_trials = residuals_r1.shape[0]

    # Helper to fit VAR on a list of paired trials and return mean F stats
    def get_mean_f_statistics(r1_data, r2_data):
        f12_list, f21_list = [], []
        for i in range(n_trials):
            # Skip if data contains NaNs
            if not np.all(np.isfinite(r1_data[i])) or not np.all(np.isfinite(r2_data[i])):
                continue

            # Stack data: column 0 = region 1, column 1 = region 2
            trial_data = np.vstack([r1_data[i], r2_data[i]]).T

            # Run VAR (Re-using your existing granger_causality function logic inline for speed)
            try:
                model = VAR(trial_data)
                res = model.fit(maxlags=max_lag_bins)

                # R1 -> R2 (Does R1 predict R2?)
                f12 = res.test_causality(caused=1, causing=[0], kind='f').test_statistic
                # R2 -> R1 (Does R2 predict R1?)
                f21 = res.test_causality(caused=0, causing=[1], kind='f').test_statistic

                f12_list.append(f12)
                f21_list.append(f21)
            except ValueError:
                # Handle cases where VAR fails to fit (e.g., constant data)
                continue

        return np.nanmean(f12_list), np.nanmean(f21_list)

    # 1. Calculate REAL Granger Causality (Matched trials: i to i)
    real_f12, real_f21 = get_mean_f_statistics(residuals_r1, residuals_r2)

    if np.isnan(real_f12) or np.isnan(real_f21):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # 2. Generate NULL Distribution (Mismatched trials: i to j)
    null_f12_dist = np.empty(n_shuffles)
    null_f21_dist = np.empty(n_shuffles)

    # We can parallelize this loop if N_CORES is high, otherwise a simple loop suffices
    # Note: Shuffling just the indices of the second region is sufficient
    for s in range(n_shuffles):
        shuffled_indices = np.random.permutation(n_trials)
        r2_shuffled = residuals_r2[shuffled_indices]

        # Compute mean F for this shuffle
        null_f12, null_f21 = get_mean_f_statistics(residuals_r1, r2_shuffled)
        null_f12_dist[s] = null_f12
        null_f21_dist[s] = null_f21

    # 3. Calculate P-values
    # Proportion of null F-stats that are greater than or equal to the real F-stat
    p_12 = (np.sum(null_f12_dist >= real_f12) + 1) / (n_shuffles + 1)
    p_21 = (np.sum(null_f21_dist >= real_f21) + 1) / (n_shuffles + 1)
    null_f12 = np.median(null_f12)
    null_f21 = np.median(null_f21)

    return real_f12, null_f12, p_12, real_f21, null_f21, p_21


def get_binned_spikes_distance(spikes, region_neurons, trials_df, d_centers):
    """
    Get spike counts for a set of neurons and trials in spatial bins.

    Args:
        spikes (dict): Spike data dictionary (must contain 'distances', 'clusters').
        region_neurons (np.array): Array of neuron IDs to include.
        trials_df (pd.DataFrame): DataFrame containing trial information (must contain 'distances').
        d_centers (np.array): Array of distance bin centers.

    Returns:
        np.array: Spike counts of shape (n_trials, n_neurons, n_bins)
    """
    n_trials = trials_df.shape[0]
    n_neurons = region_neurons.shape[0]
    n_bins = d_centers.shape[0]

    # Initialize output array
    X = np.empty((n_trials, n_neurons, n_bins))

    for b, bin_center in enumerate(d_centers):
        # Calculate intervals for this bin for all trials
        # Note: trials_df['distances'] is the start distance of the object interaction?
        # Actually in decode_context_per_object_distance.py:
        # these_intervals = np.vstack((all_obj_df['distances'] + (bin_center - (BIN_SIZE/2)),
        #                              all_obj_df['distances'] + (bin_center + (BIN_SIZE/2)))).T
        # This assumes 'distances' in trials_df is the reference point (e.g. object entry).

        these_intervals = np.vstack((trials_df['distances'] + (bin_center - (BIN_SIZE/2)),
                                     trials_df['distances'] + (bin_center + (BIN_SIZE/2)))).T

        spike_counts, neuron_ids = get_spike_counts_in_bins(spikes['distances'], spikes['clusters'],
                                                            these_intervals)

        # Align spike counts to region_neurons
        # Some neurons might have been filtered out due to speed thresholding, so we use pandas to reindex
        # and fill missing neurons with zeros.
        df_counts = pd.DataFrame(spike_counts.T, columns=neuron_ids)
        X[:, :, b] = df_counts.reindex(columns=region_neurons, fill_value=0).values

    return X


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

    # Filter spikes by speed for each probe
    for probe in spikes.keys():
        if 'speeds' in spikes[probe]:
            spikes[probe]['distances'] = spikes[probe]['distances'][spikes[probe]['speeds'] > MIN_SPEED]
            spikes[probe]['clusters'] = spikes[probe]['clusters'][spikes[probe]['speeds'] > MIN_SPEED]
        else:
            print(f"Warning: No speed data found for {probe}, skipping speed filtering.")

    # Loop over regions
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
            # Get trials for Sound 1 and Sound 2 for this object
            trials_soundA = all_obj_df.loc[(all_obj_df['object'] == this_obj) & (all_obj_df['sound'] == 1)]
            trials_soundB = all_obj_df.loc[(all_obj_df['object'] == this_obj) & (all_obj_df['sound'] == 2)]

            # Get binned spikes
            X_soundA = get_binned_spikes_distance(spikes[probe], region_neurons, trials_soundA, d_centers)
            X_soundB = get_binned_spikes_distance(spikes[probe], region_neurons, trials_soundB, d_centers)

            # Decode context per spatial bin and get decoding probabilities
            X = np.concatenate([X_soundA, X_soundB], axis=0)  # shape: (trials, neurons, bins)
            y = np.concatenate([np.zeros(X_soundA.shape[0]), np.ones(X_soundB.shape[0])]).astype(int)

            # Get trial by trial decoding probabilities and subtract the mean to leave the residuals
            trial_probs = decode_context(X, y, clf)
            subtracted_probs = trial_probs - np.tile(np.mean(trial_probs, axis=0), (trial_probs.shape[0], 1))
            residuals[f'object{this_obj}'][region] = subtracted_probs

    # Loop over region pairs
    # Create a list of jobs to run in parallel
    job_list = []

    max_lag_bins = int(MAX_LAG / STEP_SIZE)

    for region1, region2 in combinations(residuals['object1'].keys(), 2):
        for this_obj in [1, 2, 3]:
            # Get the RESIDUALS (not raw probabilities)
            # Assuming you stored residuals in a dictionary called 'residuals'
            r1_data = residuals[f'object{this_obj}'][region1]
            r2_data = residuals[f'object{this_obj}'][region2]

            job_list.append({
                'r1': region1, 'r2': region2,
                'obj': this_obj,
                'd1': r1_data, 'd2': r2_data
            })

    # Define a wrapper to unpack arguments for Parallel
    def run_pair_job(job):
        f12, nullf12, p12, f21, nullf21, p21 = compute_granger_trial_shuffling(
            job['d1'], job['d2'], max_lag_bins, n_shuffles=N_PSEUDO
        )
        return job['r1'], job['r2'], job['obj'], f12, nullf12, p12, f21, nullf21, p21

    print(f"Running Granger on {len(job_list)} pairs...")
    results = Parallel(n_jobs=N_CORES)(delayed(run_pair_job)(job) for job in job_list)

    # Unpack results into DataFrame
    for r1, r2, obj, f12, nullf12, p12, f21, nullf21, p21 in results:
        granger_df = pd.concat((granger_df, pd.DataFrame(data={
            'region_pair': [f'{r1} → {r2}', f'{r2} → {r1}'],
            'region1': [r1, r2],
            'region2': [r2, r1],
            'object': f'object{obj}',
            'p_value': [p12, p21],
            'f_stat': [f12, f21],
            'null_f_stat': [nullf12, nullf21],
            'subject': subject,
            'date': date
        })))

    # Save to disk
    granger_df.to_csv(join(path_dict['save_path'], f'granger_causality_context_{BIN_SIZE}mmbins.csv'),
                      index=False)
