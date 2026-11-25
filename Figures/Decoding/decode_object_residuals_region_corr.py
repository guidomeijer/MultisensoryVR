# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 2026

Decode object 1 vs object 2 around object entry from the time-binned GLM residuals
(output of create_time_residuals.py) with L2-regularized logistic regression, separately per
brain region and per time bin relative to object entry. The neural activity of every trial is
projected onto the weight vector of the classifier with decision_function() (raw, unbounded
linear projection), using held-out trials only (10-fold cross-validation). The trial-by-trial
projections are then correlated between all pairs of simultaneously recorded brain regions,
separately for context A and context B trials.

Two correlations are computed:
    r       : correlation of the raw projections (includes the shared object signal)
    r_noise : correlation after subtracting the mean projection per object (within context),
              i.e. only the trial-to-trial co-fluctuations around the object means

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from msvr_functions import paths, figure_style, load_objects

colors, dpi = figure_style()

# Settings
MIN_SPEED = 20  # mm/s, has to match create_time_residuals.py
BIN_SIZE = 0.1  # s, has to match create_time_residuals.py
WINDOW = [-1, 1]  # s around object entry
MIN_NEURONS = 10  # minimum number of neurons per region
N_FOLDS = 10
C = 1  # inverse of L2 regularization strength
MIN_TRIALS_CORR = 10  # minimum number of trials per context to compute a correlation

# Time grid relative to object entry (in units of bins)
rel_bins = np.arange(int(np.round(WINDOW[0] / BIN_SIZE)), int(np.round(WINDOW[1] / BIN_SIZE)) + 1)
rel_times = np.round(rel_bins * BIN_SIZE, 3)

# Load in data
path_dict = paths()
with open(path_dict['google_drive_data_path'] / f'residuals_time_{MIN_SPEED}mms.pickle',
          'rb') as handle:
    spike_dict = pickle.load(handle)

clf = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=1000)
kfold_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Group recordings by session (simultaneous recordings on different probes)
sessions = pd.DataFrame({'subject': spike_dict['subject'],
                         'date': spike_dict['date']}).astype(str).drop_duplicates()

decode_results, weight_results, corr_results = [], [], []
for subject, date in zip(sessions['subject'], sessions['date']):
    print(f'Processing {subject} {date}')

    # Find all recording indices for this session
    rec_indices = [idx for idx in range(len(spike_dict['date']))
                   if (str(spike_dict['date'][idx]) == date)
                   and (str(spike_dict['subject'][idx]) == subject)]

    # Only use time bins that are present in all probes of this session
    bin_keys = [np.round(spike_dict['time'][idx] / BIN_SIZE).astype(int) for idx in rec_indices]
    shared_keys = bin_keys[0]
    for keys in bin_keys[1:]:
        shared_keys = np.intersect1d(shared_keys, keys)
    bin_masks = [np.isin(keys, shared_keys) for keys in bin_keys]
    ref_idx, ref_mask = rec_indices[0], bin_masks[0]
    bin_times = spike_dict['time'][ref_idx][ref_mask]
    trial_per_bin = spike_dict['trial'][ref_idx][ref_mask]

    # Load in object entries (object 1 and 2 only)
    obj_df = load_objects(subject, date)
    obj_df = obj_df[obj_df['object'].isin([1, 2]) & ~np.isnan(obj_df['times'])]
    obj_df = obj_df.reset_index(drop=True)
    object_labels = obj_df['object'].values
    context_labels = obj_df['sound'].values

    # For every object entry and relative time, find the corresponding residual time bin
    # (-1 if this bin is missing, e.g. because the mouse was running too slow)
    bin_idx = np.full((obj_df.shape[0], rel_bins.shape[0]), -1)
    for e, obj_row in enumerate(obj_df.itertuples()):
        trial_bins = np.where(trial_per_bin == obj_row.trial_nr)[0]
        rel_bin = np.round((bin_times[trial_bins] - obj_row.times) / BIN_SIZE).astype(int)
        for b, this_rel_bin in zip(trial_bins, rel_bin):
            if this_rel_bin in rel_bins:
                bin_idx[e, this_rel_bin - rel_bins[0]] = b

    # Find all unique regions recorded in this session across all probes
    session_regions = []
    for idx in rec_indices:
        session_regions.extend(np.unique(spike_dict['region'][idx]))
    session_regions = [r for r in np.unique(session_regions) if r != 'root']

    # Decode object identity per region and time bin
    projections = {}  # region -> [object entries x time bins] cross-validated projection
    for region in session_regions:

        # Collect neurons for this region across probes
        region_res, region_neurons = [], []
        for idx, bin_mask in zip(rec_indices, bin_masks):
            mask = spike_dict['region'][idx] == region
            if np.sum(mask) > 0:
                region_res.append(spike_dict['residuals'][idx][bin_mask][:, mask])
                region_neurons.append(spike_dict['neuron_id'][idx][mask])
        X_all = np.hstack(region_res)
        neuron_ids = np.concatenate(region_neurons)

        # Throw out silent neurons and z-score the residuals over all time bins
        keep_neurons = np.std(X_all, axis=0) > 0
        X_all, neuron_ids = X_all[:, keep_neurons], neuron_ids[keep_neurons]
        if X_all.shape[1] < MIN_NEURONS:
            continue
        X_all = (X_all - np.mean(X_all, axis=0)) / np.std(X_all, axis=0)

        region_proj = np.full(bin_idx.shape, np.nan)
        for t, rel_time in enumerate(rel_times):

            # Get object entries for which this time bin exists
            has_bin = bin_idx[:, t] >= 0
            y = object_labels[has_bin]
            if (np.unique(y).shape[0] < 2) or (np.min(np.bincount(y)[1:]) < N_FOLDS):
                continue
            X = X_all[bin_idx[has_bin, t]]

            # Get the cross-validated projection onto the weight vector of the classifier
            proj = np.empty(y.shape[0])
            for train_index, test_index in kfold_cv.split(X, y):
                clf.fit(X[train_index], y[train_index])
                proj[test_index] = clf.decision_function(X[test_index])
            region_proj[has_bin, t] = proj

            # Decoding accuracy (positive projection = object 2)
            pred = np.where(proj > 0, 2, 1)
            decode_results.append({
                'time': rel_time, 'accuracy': np.mean(pred == y), 'n_trials': y.shape[0],
                'n_neurons': X.shape[1], 'region': region, 'subject': subject, 'date': date})

            # Neural weights of the classifier fit on all trials
            clf.fit(X, y)
            weight_results.append({
                'time': rel_time, 'region': region, 'subject': subject, 'date': date,
                'neuron_id': neuron_ids, 'weights': clf.coef_[0].copy(),
                'intercept': clf.intercept_[0]})

        projections[region] = region_proj

    # Correlate the projections trial-by-trial between pairs of regions, per context
    for region1, region2 in combinations(projections.keys(), 2):
        for context in [1, 2]:
            for t, rel_time in enumerate(rel_times):
                proj1 = projections[region1][:, t]
                proj2 = projections[region2][:, t]
                use = (context_labels == context) & ~np.isnan(proj1) & ~np.isnan(proj2)
                if np.sum(use) < MIN_TRIALS_CORR:
                    continue
                p1, p2, objs = proj1[use], proj2[use], object_labels[use]

                # Raw correlation
                r = np.corrcoef(p1, p2)[0, 1]

                # Noise correlation: subtract the mean projection per object
                p1_res, p2_res = p1.copy(), p2.copy()
                for obj in [1, 2]:
                    p1_res[objs == obj] -= np.mean(p1[objs == obj])
                    p2_res[objs == obj] -= np.mean(p2[objs == obj])
                r_noise = np.corrcoef(p1_res, p2_res)[0, 1]

                corr_results.append({
                    'time': rel_time, 'context': context, 'r': r, 'r_noise': r_noise,
                    'n_trials': np.sum(use), 'region1': region1, 'region2': region2,
                    'region_pair': f'{region1}-{region2}', 'subject': subject, 'date': date})

decode_df = pd.DataFrame(decode_results)
weights_df = pd.DataFrame(weight_results)
corr_df = pd.DataFrame(corr_results)
corr_df['context'] = corr_df['context'].map({1: 'Context A', 2: 'Context B'})

# Save to disk
#decode_df.to_csv(path_dict['save_path'] / 'decode_object_residuals.csv', index=False)
#corr_df.to_csv(path_dict['save_path'] / 'decode_object_residuals_region_corr.csv', index=False)
#weights_df.to_pickle(path_dict['save_path'] / 'decode_object_residuals_weights.pickle')

# %% Plot decoding accuracy per region

regions = sorted(decode_df['region'].unique())
n_cols = 4
n_rows = int(np.ceil(len(regions) / n_cols))
f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.75, n_rows * 1.75), dpi=dpi,
                      sharex=True, squeeze=False)
axs = axs.flatten()
for r, region in enumerate(regions):
    ax = axs[r]
    plot_df = decode_df[decode_df['region'] == region]
    sns.lineplot(data=plot_df, x='time', y='accuracy', ax=ax, color='k',
                 errorbar='se', err_kws={'lw': 0})
    ax.axhline(0.5, ls='--', lw=0.5, color='grey', zorder=0)
    ax.axvline(0, ls='--', lw=0.5, color='k', zorder=0)
    n_rec = plot_df[['subject', 'date']].drop_duplicates().shape[0]
    ax.set(title=f'{region} (n={n_rec})', xlabel='', ylabel='')
    if r % n_cols == 0:
        ax.set_ylabel('Object decoding accuracy')
for ax in axs[len(regions):]:
    ax.axis('off')
f.supxlabel('Time from object entry (s)', fontsize=7)
sns.despine(trim=True)
plt.tight_layout()
#plt.savefig(path_dict['fig_path'] / 'decode_object_residuals_accuracy.pdf')
plt.show()

# %% Plot correlation between region pairs per context

for corr_type, ylabel in zip(['r', 'r_noise'], ['Projection correlation (r)',
                                                 'Projection noise correlation (r)']):
    region_pairs = sorted(corr_df['region_pair'].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(region_pairs) / n_cols))
    f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.75, n_rows * 1.75), dpi=dpi,
                          sharex=True, squeeze=False)
    axs = axs.flatten()
    for p, region_pair in enumerate(region_pairs):
        ax = axs[p]
        plot_df = corr_df[corr_df['region_pair'] == region_pair]
        sns.lineplot(data=plot_df, x='time', y=corr_type, hue='context', ax=ax,
                     hue_order=['Context A', 'Context B'],
                     palette=[colors['context1'], colors['context2']],
                     errorbar='se', err_kws={'lw': 0}, legend=(p == 0))
        ax.axhline(0, ls='--', lw=0.5, color='grey', zorder=0)
        ax.axvline(0, ls='--', lw=0.5, color='k', zorder=0)
        n_rec = plot_df[['subject', 'date']].drop_duplicates().shape[0]
        ax.set(title=f'{region_pair} (n={n_rec})', xlabel='', ylabel='')
        if p % n_cols == 0:
            ax.set_ylabel(ylabel)
    if axs[0].get_legend() is not None:
        axs[0].legend(title='')
    for ax in axs[len(region_pairs):]:
        ax.axis('off')
    f.supxlabel('Time from object entry (s)', fontsize=7)
    sns.despine(trim=True)
    plt.tight_layout()
    #plt.savefig(path_dict['fig_path'] / f'decode_object_residuals_{corr_type}.pdf')
    plt.show()
