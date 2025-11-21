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
clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=500))
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


def process_trial_granger(trial_idx, prob_dict, r1, r2, n_shuffles):
    """
    Performs Granger causality using a time-shifting surrogate method.
    """
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

    # Calculate max lag in bins
    max_lag_bins = int(MAX_LAG / STEP_SIZE)

    for i in range(n_shuffles):
        # Generate a random shift. It must not be 0.
        # Ensure shift is > max_lag to break short-term relationships.
        shift = np.random.randint(max_lag_bins + 1, n_timepoints - (max_lag_bins + 1))
        
        # Shift the SECOND time series
        shifted_ts_r2 = np.roll(real_ts_r2, shift)
        
        # Now create the surrogate pair using the original r1 and the shifted r2
        surrogate_data = np.vstack([real_ts_r1, shifted_ts_r2]).T
        reg1_reg2_shuf[i], reg2_reg1_shuf[i] = granger_causality(surrogate_data)

    # 3. Calculate p-values
    p_12 = (np.sum(reg1_reg2_shuf >= f_12) + 1) / (n_shuffles + 1)
    p_21 = (np.sum(reg2_reg1_shuf >= f_21) + 1) / (n_shuffles + 1)
    
    return f_12, f_21, p_12, p_21


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
    
    prob, prob['object1'], prob['object2'], prob['object3'] = dict(), dict(), dict(), dict()
    
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
            
            prob[f'object{this_obj}'][region] = decode_context(X, y, clf)            
        
    # Do Granger causality for all region pairs
    for region1, region2 in combinations(prob['object1'].keys(), 2):
        print(f'Running Granger causality for {region1} → {region2}')
        for this_obj in [1, 2, 3]:
            n_trials = prob[f'object{this_obj}'][region1].shape[0]
            
            # Use joblib to parallelize the loop over trials
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
            valid_trials = ~np.isnan(reg1_reg2_p)
            if np.sum(valid_trials) > 0:
                p_reg1_reg2 = stats.binomtest(np.sum(reg1_reg2_p[valid_trials] < 0.05), 
                                              np.sum(valid_trials), 0.05).pvalue
            else:
                p_reg1_reg2 = np.nan
                                          
            valid_trials = ~np.isnan(reg2_reg1_p)
            if np.sum(valid_trials) > 0:
                p_reg2_reg1 = stats.binomtest(np.sum(reg2_reg1_p[valid_trials] < 0.05), 
                                              np.sum(valid_trials), 0.05).pvalue
            else:
                p_reg2_reg1 = np.nan
                
            # Add to dataframe
            granger_df = pd.concat((granger_df, pd.DataFrame(data={
                'region_pair': [f'{region1} → {region2}', f'{region2} → {region1}'],
                'region1': [region1, region2], 'region2': [region2, region1],
                'object': f'object{this_obj}',
                'p_value': [p_reg1_reg2, p_reg2_reg1],
                'f_stat': [np.nanmean(reg1_reg2), np.nanmean(reg2_reg1)],
                'subject': subject, 'date': date})))
        
    # Save to disk
    granger_df.to_csv(join(path_dict['save_path'], 'granger_causality_context_distance.csv'), index=False)
