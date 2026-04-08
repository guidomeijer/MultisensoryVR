# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""

import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from msvr_functions import (paths, load_multiple_probes, load_subjects, load_objects,
                            get_spike_counts_in_bins)

# Settings
D_BEFORE = 300  # mm
D_AFTER = 300
BIN_SIZE = 20
STEP_SIZE = BIN_SIZE  # Set step size equal to bin size for non-overlapping bins
MIN_NEURONS = 10
MIN_TRIALS = 30
MIN_SPEED = 20  # mm/s
N_CORES = 18
MAX_LAG = 100  # mm 
N_PSEUDO = 500
OVERWRITE = True

# Create distance array
d_centers = np.arange(-D_BEFORE + (BIN_SIZE/2), D_AFTER, BIN_SIZE)

# Initialize
clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))

if OVERWRITE:
    granger_df = pd.DataFrame()
else:
    granger_df = pd.read_csv(path_dict['save_path'] / f'granger_causality_context_{BIN_SIZE}mmbins.csv')
    granger_df[['subject', 'date']] = granger_df[['subject', 'date']].astype(str)
    merged = rec.merge(granger_df, on=['subject', 'date'], how='left', indicator=True)
    rec = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

# Functions for parallelization
def decode_context(X, y, clf):
    """
    Decodes object identity per spatial bin and returns the decision variable 
    (evidence for the true class) using leave-one-out cross-validation.

    Args:
        X (np.array): Data of shape (n_trials, n_neurons, n_bins)
        y (np.array): Labels of shape (n_trials,)
        clf: Scikit-learn classifier instance.

    Returns:
        np.array: Decoding decision variables of shape (n_trials, n_bins)
    """
    decoding_vals = np.empty((X.shape[0], X.shape[2]))
    loo = LeaveOneOut()

    # Get unique classes to correctly index decision variables
    classes = np.unique(y)

    for tb in range(X.shape[2]):
        X_t = X[:, :, tb] # shape: (n_trials, n_neurons)

        # Get decision variables using cross-validation
        all_decs = cross_val_predict(clf, X_t, y, cv=loo, method='decision_function', n_jobs=N_CORES)

        if all_decs.ndim == 1:
            # Binary case: decision_function returns a single column for the higher-indexed class
            all_decs_cols = np.stack([-all_decs, all_decs], axis=1)
        else:
            all_decs_cols = all_decs

        # Select the value corresponding to the *true* label for each trial
        true_class_indices = np.searchsorted(classes, y)
        decoding_vals[:, tb] = all_decs_cols[np.arange(len(y)), true_class_indices]

    return decoding_vals


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
    n_bins = residuals_r1.shape[1]
    L = max_lag_bins
    valid_obs = n_bins - L

    def get_joint_f_statistics(r1_data, r2_data):
        # Construct joint design matrices across all trials
        Y1 = np.full(n_trials * valid_obs, np.nan)
        Y2 = np.full(n_trials * valid_obs, np.nan)
        # X contains: [Intercept, Lags of R1 (L columns), Lags of R2 (L columns)]
        X = np.full((n_trials * valid_obs, 2 * L + 1), np.nan)
        X[:, 0] = 1  # Constant term

        for i in range(n_trials):
            start, end = i * valid_obs, (i + 1) * valid_obs
            y1, y2 = r1_data[i], r2_data[i]
            
            Y1[start:end] = y1[L:]
            Y2[start:end] = y2[L:]
            
            for l in range(L):
                X[start:end, 1 + l] = y1[L - 1 - l : n_bins - 1 - l]
                X[start:end, 1 + L + l] = y2[L - 1 - l : n_bins - 1 - l]

        # Remove observations with NaNs
        mask = ~np.any(np.isnan(np.column_stack([Y1, Y2, X])), axis=1)
        if np.sum(mask) < (2 * L + 1): 
            return np.nan, np.nan
        
        Y1_clean, Y2_clean, X_clean = Y1[mask], Y2[mask], X[mask]

        def calculate_f_stat(Y, X_full, restrict_cols):
            # Fit full model and calculate Sum of Squared Residuals (SSR)
            beta_full, ssr_full, _, _ = np.linalg.lstsq(X_full, Y, rcond=None)
            if len(ssr_full) == 0: ssr_full = [np.sum((Y - X_full @ beta_full)**2)]
            
            # Fit restricted model (omitting the lags of the 'causing' variable)
            X_red = np.delete(X_full, restrict_cols, axis=1)
            beta_red, ssr_red, _, _ = np.linalg.lstsq(X_red, Y, rcond=None)
            if len(ssr_red) == 0: ssr_red = [np.sum((Y - X_red @ beta_red)**2)]
            
            N, K = X_full.shape
            q = len(restrict_cols)
            return ((ssr_red[0] - ssr_full[0]) / q) / (ssr_full[0] / (N - K))

        # R1 -> R2: Test if lags of R1 (cols 1 to L) improve prediction of R2
        f12 = calculate_f_stat(Y2_clean, X_clean, np.arange(1, L + 1))
        # R2 -> R1: Test if lags of R2 (cols L+1 to 2L) improve prediction of R1
        f21 = calculate_f_stat(Y1_clean, X_clean, np.arange(L + 1, 2 * L + 1))
        return f12, f21

    # 1. Calculate REAL Granger Causality (Matched trials: i to i)
    real_f12, real_f21 = get_joint_f_statistics(residuals_r1, residuals_r2)

    if np.isnan(real_f12) or np.isnan(real_f21):
        return [np.nan] * 6

    # 2. Generate NULL Distribution (Mismatched trials: i to j)
    null_f12_dist = np.empty(n_shuffles)
    null_f21_dist = np.empty(n_shuffles)

    # We can parallelize this loop if N_CORES is high, otherwise a simple loop suffices
    # Note: Shuffling just the indices of the second region is sufficient
    for s in range(n_shuffles):
        shuffled_indices = np.random.permutation(n_trials)
        r2_shuffled = residuals_r2[shuffled_indices]

        # Compute mean F for this shuffle
        shuff_f12, shuff_f21 = get_joint_f_statistics(residuals_r1, r2_shuffled)
        null_f12_dist[s] = shuff_f12
        null_f21_dist[s] = shuff_f21

    # 3. Calculate P-values
    # Proportion of null F-stats that are greater than or equal to the real F-stat
    p_12 = (np.sum(null_f12_dist >= real_f12) + 1) / (n_shuffles + 1)
    p_21 = (np.sum(null_f21_dist >= real_f21) + 1) / (n_shuffles + 1)
    null_f12 = np.nanmedian(null_f12_dist)
    null_f21 = np.nanmedian(null_f21_dist)

    return real_f12, null_f12, p_12, real_f21, null_f21, p_21


def get_binned_spikes_distance(spikes, region_neurons, trials_df, d_centers, behave_dist, behave_times):
    """
    Get spike counts for a set of neurons and trials in spatial bins,
    normalized by time spent in each bin to get firing rates (Hz).

    Args:
        spikes (dict): Spike data dictionary (must contain 'distances', 'clusters').
        region_neurons (np.array): Array of neuron IDs to include.
        trials_df (pd.DataFrame): DataFrame containing trial information (must contain 'distances').
        d_centers (np.array): Array of distance bin centers.
        behave_dist (np.array): Distance array from behavioral data.
        behave_times (np.array): Time array from behavioral data.

    Returns:
        np.array: Firing rates of shape (n_trials, n_neurons, n_bins)
    """
    n_trials = trials_df.shape[0]
    n_neurons = region_neurons.shape[0]
    n_bins = d_centers.shape[0]

    # Initialize output array
    X = np.empty((n_trials, n_neurons, n_bins))

    # Calculate median time step to convert sample counts to seconds
    dt = np.nanmedian(np.diff(behave_times))

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

        # Calculate time spent in each interval for each trial (occupancy)
        bin_times = np.zeros(n_trials)
        for t in range(n_trials):
            mask = (behave_dist >= these_intervals[t, 0]) & (behave_dist <= these_intervals[t, 1])
            bin_times[t] = np.sum(mask) * dt

        # Align spike counts to region_neurons
        # Some neurons might have been filtered out due to speed thresholding, so we use pandas to reindex
        # and fill missing neurons with zeros.
        df_counts = pd.DataFrame(spike_counts.T, columns=neuron_ids)
        counts_aligned = df_counts.reindex(columns=region_neurons, fill_value=0).values

        # Normalize counts by time spent to get firing rate (Hz)
        bin_times[bin_times == 0] = np.nan  # Avoid division by zero
        X[:, :, b] = counts_aligned / bin_times[:, np.newaxis]

    return X


# %% Loop over recordings
for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'\n{subject} {date} ({i} of {rec.shape[0]})')

    # Load in data
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_multiple_probes(session_path)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)

    # Load and filter behavior data by speed
    behave = pd.read_csv(join(session_path, 'behave.csv'))
    if 'speed' in behave.columns:
        behave = behave[behave['speed'] > MIN_SPEED]

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
            X_soundA = get_binned_spikes_distance(spikes[probe], region_neurons, trials_soundA, d_centers,
                                                   behave['distance'].values, behave['time'].values)
            X_soundB = get_binned_spikes_distance(spikes[probe], region_neurons, trials_soundB, d_centers,
                                                   behave['distance'].values, behave['time'].values)

            # Decode context per spatial bin and get decoding probabilities
            X = np.concatenate([X_soundA, X_soundB], axis=0)  # shape: (trials, neurons, bins)
            y = np.concatenate([np.zeros(X_soundA.shape[0]), np.ones(X_soundB.shape[0])]).astype(int)

            # Get trial by trial decoding decision variables and subtract the mean to leave the residuals
            trial_decs = decode_context(X, y, clf)
            subtracted_decs = trial_decs - np.tile(np.mean(trial_decs, axis=0), (trial_decs.shape[0], 1))
            residuals[f'object{this_obj}'][region] = subtracted_decs

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
