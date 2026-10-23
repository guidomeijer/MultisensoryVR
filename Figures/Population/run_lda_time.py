# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 2026

Fit a single LDA axis per brain region that separates the two contexts using the time-binned
GLM residuals of all simultaneously recorded neurons (output of create_time_residuals.py).
Every time bin is projected onto the axis and its signed distance to the decision boundary is
taken around the entries into object 1 and object 2. Context separability is the absolute
distance to the decision boundary (higher = further from the boundary), averaged over trials
with both contexts weighted equally.

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from msvr_functions import paths, load_subjects, figure_style, load_objects

colors, dpi = figure_style()

# Settings
MIN_NEURONS = 5
BIN_SIZE = 0.1  # s, has to match create_time_residuals.py
WINDOW = [-2, 2]  # s around object entry
CROSS_VALIDATE = True  # if True, project held-out trials only (folds grouped by trial)
N_FOLDS = 5

# Load in data
path_dict = paths()
subjects = load_subjects()
with open(path_dict['google_drive_data_path'] / 'residuals_time_20mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Group recordings by date (simultaneous recordings)
unique_dates = np.unique(spike_dict['date'])

lda_results = []
for date in unique_dates:
    print(f'Processing {date}')

    # Find all recording indices for this date
    rec_indices = [idx for idx in range(len(spike_dict['date'])) if spike_dict['date'][idx] == date]
    this_subject = spike_dict['subject'][rec_indices[0]]
    is_far = subjects.loc[subjects['SubjectID'] == this_subject, 'Far'].values[0]

    # Load in object data
    obj_df = load_objects(this_subject, str(date))

    # Only use time bins that are present in all probes of this session
    bin_keys = [np.round(spike_dict['time'][idx] / BIN_SIZE).astype(int) for idx in rec_indices]
    shared_keys = bin_keys[0]
    for keys in bin_keys[1:]:
        shared_keys = np.intersect1d(shared_keys, keys)
    bin_masks = [np.isin(keys, shared_keys) for keys in bin_keys]
    ref_idx, ref_mask = rec_indices[0], bin_masks[0]
    bin_times = spike_dict['time'][ref_idx][ref_mask]
    trial_per_bin = spike_dict['trial'][ref_idx][ref_mask]
    context_per_bin = spike_dict['context'][ref_idx][ref_mask]
    y = (context_per_bin == 1).astype(int)
    if np.unique(y).shape[0] < 2:
        continue

    # Get time bins around entries of object 1 and 2 (relative time on the bin grid)
    peri_bins = []
    for obj_row in obj_df[obj_df['object'].isin([1, 2])].itertuples():
        if np.isnan(obj_row.times):
            continue
        rel_time = bin_times - obj_row.times
        in_win = ((trial_per_bin == obj_row.trial_nr)
                  & (rel_time >= WINDOW[0]) & (rel_time <= WINDOW[1]))
        for b in np.where(in_win)[0]:
            peri_bins.append({'bin': b, 'object': obj_row.object,
                              'time': np.round(rel_time[b] / BIN_SIZE) * BIN_SIZE})
    if len(peri_bins) == 0:
        continue
    peri_bins = pd.DataFrame(peri_bins)

    # Find all unique regions recorded on this date across all probes
    session_regions = []
    for idx in rec_indices:
        session_regions.extend(np.unique(spike_dict['region'][idx]))
    session_regions = [r for r in np.unique(session_regions) if r != 'root']

    for region in session_regions:

        # Collect neurons for this region across probes
        region_spikes = []
        for idx, bin_mask in zip(rec_indices, bin_masks):
            mask = spike_dict['region'][idx] == region
            if np.sum(mask) > 0:
                region_spikes.append(spike_dict['residuals'][idx][bin_mask][:, mask])
        if len(region_spikes) == 0:
            continue
        X = np.hstack(region_spikes)

        # Throw out silent neurons
        X = X[:, np.std(X, axis=0) > 0]
        if X.shape[1] < MIN_NEURONS:
            continue

        # Z-score the residuals
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Fit LDA on all time bins and get the signed distance to the decision boundary
        lda_distance = np.empty(X.shape[0])
        if CROSS_VALIDATE:
            splits = GroupKFold(n_splits=N_FOLDS).split(X, y, groups=trial_per_bin)
        else:
            splits = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]
        for train_idx, test_idx in splits:
            lda = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
            lda.fit(X[train_idx], y[train_idx])
            lda_distance[test_idx] = (lda.decision_function(X[test_idx])
                                      / np.linalg.norm(lda.coef_))

        # Add to results
        this_df = peri_bins.copy()
        this_df['lda_distance'] = lda_distance[this_df['bin'].values]
        this_df['lda_distance_abs'] = np.abs(this_df['lda_distance'])
        this_df['trial'] = trial_per_bin[this_df['bin'].values]
        this_df['context'] = context_per_bin[this_df['bin'].values]
        this_df['region'] = region
        this_df['n_neurons'] = X.shape[1]
        this_df['subject'] = this_subject
        this_df['date'] = date
        this_df['is_far'] = is_far
        lda_results.append(this_df.drop(columns='bin'))

lda_df = pd.concat(lda_results, ignore_index=True)

# %% Plot

# Per recording, average the absolute distance over trials within each context and then over
# the two contexts, so that recordings are the unit of the error bars
per_rec_df = (lda_df.groupby(['subject', 'date', 'region', 'object', 'time', 'context'])
              ['lda_distance_abs'].mean()
              .groupby(['subject', 'date', 'region', 'object', 'time']).mean()
              .reset_index())
per_rec_df['object'] = per_rec_df['object'].map({1: 'Object 1', 2: 'Object 2'})

regions = sorted(per_rec_df['region'].unique())
n_cols = 4
n_rows = int(np.ceil(len(regions) / n_cols))
f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.75, n_rows * 1.75), dpi=dpi,
                      sharex=True, squeeze=False)
axs = axs.flatten()
for r, region in enumerate(regions):
    ax = axs[r]
    plot_df = per_rec_df[per_rec_df['region'] == region]
    sns.lineplot(data=plot_df, x='time', y='lda_distance_abs', hue='object', ax=ax,
                 hue_order=['Object 1', 'Object 2'], palette=[colors['obj1'], colors['obj2']],
                 errorbar='se', err_kws={'lw': 0}, legend=(r == 0))
    ax.axvline(0, ls='--', lw=0.5, color='k', zorder=0)
    n_rec = plot_df[['subject', 'date']].drop_duplicates().shape[0]
    ax.set(title=f'{region} (n={n_rec})', xlabel='', ylabel='')
    if r % n_cols == 0:
        ax.set_ylabel('Distance to boundary (LDA)')
if axs[0].get_legend() is not None:
    axs[0].legend(title='')
for ax in axs[len(regions):]:
    ax.axis('off')
f.supxlabel('Time from object entry (s)', fontsize=7)
sns.despine(trim=True)
plt.tight_layout()
plt.show()
