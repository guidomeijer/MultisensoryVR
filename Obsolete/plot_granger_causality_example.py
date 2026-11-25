# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from scipy import stats
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.special import logit
import matplotlib.pyplot as plt
import seaborn as sns
from msvr_functions import (paths, load_multiple_probes, load_subjects, load_objects,
                            calculate_peths, figure_style)
colors, dpi = figure_style()

# Settings
T_BEFORE = 2  # s
T_AFTER = 1
BIN_SIZE = 0.05
MIN_NEURONS = 10
MIN_TRIALS = 20
N_CORES = 20
MAX_LAG = 0.5  # s
#SUBJECT = '459601'
#SESSION = '20240411'
SUBJECT = '478154'
SESSION = '20251008'

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))

clf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=500, max_depth=5)

# Functions for parallelization
def decode_stationary_context(this_x, this_y, this_clf, n_splits=10, n_timebins=60):
    n_trials, n_neurons, _ = this_x.shape
    decoding_probs = np.empty((n_trials, n_timebins))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(this_x, this_y):

        # Create training set by pooling together all time bins
        X_train_pooled = np.transpose(this_x[train_idx], (0, 2, 1)).reshape(-1, n_neurons)
        y_train_pooled = np.repeat(this_y[train_idx], n_timebins)

        # Train classifier
        this_clf.fit(X_train_pooled, y_train_pooled)

        # Predict on test trials
        X_test_pooled = np.transpose(X[test_idx], (0, 2, 1)).reshape(-1, n_neurons)
        probs_pooled = this_clf.predict_proba(X_test_pooled)
        probs_reshaped = probs_pooled.reshape(len(test_idx), n_timebins, -1)

        classes = this_clf.classes_
        true_class_indices = np.searchsorted(classes, y[test_idx])

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
        float: F-statistic for Source -> Target causality.
    """
    n_trials, n_timebins = target_trials.shape

    if n_timebins <= (lags * 2 + 1):
        return np.nan

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
            return np.nan

        # Pooled Degrees of Freedom
        n_total_observations = len(Y_pooled)
        df_num = lags
        df_denom = n_total_observations - (1 + 2 * lags)

        f_stat = ((rss_res[0] - rss_full[0]) / df_num) / (rss_full[0] / df_denom)
        f_stat = max(0, f_stat)
        p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)

        return f_stat, p_value

    except np.linalg.LinAlgError:
        return np.nan


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

prob, prob_logit, residuals = dict(), dict(), dict()
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
        T_BEFORE, T_AFTER, BIN_SIZE, smoothing=0
    )
    _, no_goal_spikes = calculate_peths(
        spikes[probe]['times'], spikes[probe]['clusters'], region_neurons,
        all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 0), 'times'].values,
        T_BEFORE, T_AFTER, BIN_SIZE, smoothing=0
    )

    # Decode context per timebin and get decoding probabilities
    X = np.concatenate([goal_spikes, no_goal_spikes], axis=0)  # shape: (trials, neurons, time)
    y = np.concatenate([np.ones(goal_spikes.shape[0]), np.zeros(no_goal_spikes.shape[0])]).astype(int)

    # Get trial by trial decoding probabilities and subtract the mean to leave the residuals
    prob[region] = decode_stationary_context(X, y, clf)

    # Do logit transformation of probabilities
    prob_logit[region] = logit(prob[region])

    # Subtract mean to get residuals
    residuals[region] = np.zeros_like(prob_logit[region])
    residuals[region][y == 1, :] = prob_logit[region][y == 1, :] - np.mean(prob_logit[region][y == 1, :], axis=0)
    residuals[region][y == 0, :] = prob_logit[region][y == 0, :] - np.mean(prob_logit[region][y == 0, :], axis=0)

# Do Granger causality for all region pairs on the whole pooled dataset
print('Run Pooled Granger causality..')
f_12, f_21, p_12, p_21 = dict(), dict(), dict(), dict()
max_lag_bins = int(MAX_LAG / BIN_SIZE)

for region1, region2 in combinations(residuals.keys(), 2):
    # Single call per pair, passing the entire (n_trials, n_timebins) array
    f_12[f'{region1}-{region2}'], p_12[f'{region1}-{region2}'] = pooled_bivariate_gc_f(
        residuals[region2], residuals[region1], max_lag_bins
    )
    f_21[f'{region1}-{region2}'], p_12[f'{region1}-{region2}'] = pooled_bivariate_gc_f(
        residuals[region1], residuals[region2], max_lag_bins
    )

# %% Plot

plot_trial = 4
plot_region1 = 'AUD'
plot_region2 = 'VIS'

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.5*3, 1.75), dpi=dpi)
ax1.plot(tscale['tscale'], np.mean(prob[plot_region2][y == 1, :], axis=0),
         label=plot_region2, color=colors[plot_region2], lw=2)
ax1.plot(tscale['tscale'], prob[plot_region2][plot_trial, :], label=plot_region2, color='grey')
ax1.set(xlabel='Time from object entry (s)', ylabel='Decoding probability (%)',
        xticks=[-2, -1, 0, 1])

ax2.plot(tscale['tscale'], residuals[plot_region2][plot_trial, :], label=plot_region2, color=colors[plot_region2])
ax2.set(xlabel='Time from object entry (s)', ylabel='Residuals (a.u.)',
        xticks=[-2, -1, 0, 1])

ax3.plot(tscale['tscale'], residuals[plot_region1][plot_trial, :], label=plot_region1, color=colors[plot_region1])
ax3.plot(tscale['tscale'], residuals[plot_region2][plot_trial, :], label=plot_region2, color=colors[plot_region2])
ax3.set(xlabel='Time from object entry (s)', ylabel='Residuals (a.u.)', xticks=[-2, -1, 0, 1],
        title=f'{plot_region1} → {plot_region2}: {np.round(f_12[f"{plot_region1}-{plot_region2}"], 1)}'
              f'\n{plot_region1} ← {plot_region2}: {np.round(f_21[f"{plot_region1}-{plot_region2}"], 1)}')
ax3.legend()

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'granger_example.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'granger_example.pdf')
plt.show()