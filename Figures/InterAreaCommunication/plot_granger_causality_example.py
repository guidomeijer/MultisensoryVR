# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from msvr_functions import (paths, load_multiple_probes, load_subjects, load_objects,
                            calculate_peths, figure_style)
colors, dpi = figure_style()

# Settings
T_BEFORE = 2  # s
T_AFTER = 1
BIN_SIZE = 0.05
SMOOTHING = 0.1
MIN_NEURONS = 10
MIN_TRIALS = 30
N_CORES = 6
MAX_LAG = 0.5  # s
SUBJECT = '459601'
SESSION = '20240411'

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=1))


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
        X_t = X[:, :, tb]  # shape: (n_trials, n_neurons)

        # Get probabilities for all classes using cross-validation
        all_probs = cross_val_predict(clf, X_t, y, cv=loo, method='predict_proba', n_jobs=N_CORES)

        # Select the probability corresponding to the *true* label for each trial
        true_class_indices = np.searchsorted(classes, y)
        decoding_probs[:, tb] = all_probs[np.arange(len(y)), true_class_indices]

    return decoding_probs


def fast_bivariate_gc_f(target, source, lags):
    """
    Fast bivariate Granger Causality F-statistic calculation using OLS.
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
        return max(0, f_stat) 
    except:
        return np.nan


def run_gc_trial(trial, res1, res2, lags):
    f12 = fast_bivariate_gc_f(res2[trial], res1[trial], lags)
    f21 = fast_bivariate_gc_f(res1[trial], res2[trial], lags)
    return f12, f21


# Load in data
session_path = join(path_dict['local_data_path'], 'subjects', f'{SUBJECT}', f'{SESSION}')
spikes, clusters, channels = load_multiple_probes(session_path)
trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', SUBJECT, SESSION, 'trials.csv'))
all_obj_df = load_objects(SUBJECT, SESSION)


# %% Loop over regions

# Get list of all regions and which probe they were recorded on
regions, region_probes = [], []
for p, probe in enumerate(spikes.keys()):
    regions.append(np.unique(clusters[probe]['region']))
    region_probes.append([probe] * np.unique(clusters[probe]['region']).shape[0])
regions = np.concatenate(regions)
region_probes = np.concatenate(region_probes)

prob, residuals = dict(), dict()
for r, (region, probe) in enumerate(zip(regions, region_probes)):
    if region == 'root':
        continue
    print(f'Decoding {region}')
    
    # Get region neurons
    region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
    if region_neurons.shape[0] < MIN_NEURONS:
        continue

    # Context at object 2
    tscale, goal_spikes = calculate_peths(
        spikes[probe]['times'], spikes[probe]['clusters'], region_neurons,
        all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 1), 'times'].values,
        T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
    )
    _, no_goal_spikes = calculate_peths(
        spikes[probe]['times'], spikes[probe]['clusters'], region_neurons,
        all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 0), 'times'].values,
        T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING
    )

    # Decode context per timebin and get decoding probabilities
    X = np.concatenate([goal_spikes, no_goal_spikes], axis=0)  # shape: (trials, neurons, time)
    y = np.concatenate([np.zeros(goal_spikes.shape[0]), np.ones(no_goal_spikes.shape[0])]).astype(int)

    # Get trial by trial decoding probabilities and subtract the mean to leave the residuals
    prob[region] = decode_context(X, y, clf)
    residuals[region] = prob[region] - np.mean(prob[region], axis=0)

# Do Granger causality for all region pairs
print('Run Granger causality..')
f_12, f_21 = dict(), dict()
max_lag_bins = int(MAX_LAG / BIN_SIZE)

for region1, region2 in combinations(residuals.keys(), 2):
    # Do Granger causality per trial
    n_trials = residuals[region1].shape[0]
    results = Parallel(n_jobs=N_CORES)(delayed(run_gc_trial)(
        trial, residuals[region1], residuals[region2], max_lag_bins) for trial in range(n_trials))
    
    f_12[f'{region1}-{region2}'] = np.array([i[0] for i in results])
    f_21[f'{region1}-{region2}'] = np.array([i[1] for i in results])

# %% Plot

plot_trial = 36
plot_region1 = 'LEC'
plot_region2 = 'CA1'

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.5*2, 1.75), dpi=dpi)
ax1.plot(tscale['tscale'], np.mean(prob[plot_region2], axis=0), label=plot_region2, color=colors[plot_region2],
         lw=2)
ax1.plot(tscale['tscale'], prob[plot_region2][plot_trial, :], label=plot_region2, color='grey')
ax1.set(xlabel='Time from object entry (s)', ylabel='Decoding probability (%)', yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        xticks=[-2, -1, 0, 1])

ax2.plot(tscale['tscale'], residuals[plot_region2][plot_trial, :], label=plot_region2, color=colors[plot_region2])
ax2.set(xlabel='Time from object entry (s)', ylabel='Residuals (%)', yticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
        xticks=[-2, -1, 0, 1])

sns.despine(trim=True)
plt.tight_layout()
#plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'granger_example.jpg', dpi=600)
#plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'granger_example.pdf')
plt.show()

# %%
f, ax1 = plt.subplots(1, 1, figsize=(1.75, 1.9), dpi=dpi)
ax1.plot(tscale['tscale'], prob[plot_region1][plot_trial, :], label=plot_region1, color=colors[plot_region1])
ax1.plot(tscale['tscale'], prob[plot_region2][plot_trial, :], label=plot_region2, color=colors[plot_region2])
ax1.set(xlabel='Time from object entry (s)', ylabel='Decoding probability (%)', yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        title=f'{plot_region1} → {plot_region2}: {np.round(f_12[f"{plot_region1}-{plot_region2}"][plot_trial], 1)}'
              f'\n{plot_region1} ← {plot_region2}: {np.round(f_21[f"{plot_region1}-{plot_region2}"][plot_trial], 1)}')
ax1.legend()

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'granger_example.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'granger_example.pdf')
plt.show()