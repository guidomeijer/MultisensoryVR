# %% -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 2026

Same as region_corr_lda.py, but with a fixed LDA context axis and pooled noise correlations.

1) Fixed axis per epoch: the time around object entry is split into epochs (AXIS_WINDOWS, by
   default -1 to 0 s and 0 to 1 s). Per session, brain region, object and epoch, a single LDA
   axis that separates context A from context B is fit on the time-binned GLM residuals (output
   of create_time_residuals.py) of all time bins in that epoch (every bin is a sample). The
   activity of every time bin in the epoch is then projected onto this same axis, so within an
   epoch the time course of the projection reflects changes in the data and not changes in the
   axis. A time bin belongs to the epoch that contains its center; epochs exclude their start
   and include their end, except the first epoch which includes its start (so t = 0 belongs to
   the -1 to 0 s epoch). The axis is cross-validated over object entries (10-fold): held-out entries
   are projected onto the axis fit on the other entries. The projection is onto the
   unit-length axis, centered on the midpoint between the training class means
   (positive = context B).

2) Pooled correlations: per region pair and time bin, the projections are z-scored within every
   object x context group (using only the entries that have a projection in both regions), which
   removes the context and object signal. The z-scored projections of all groups are then
   concatenated and correlated in one go. With POOL_OBJECTS = True both objects are pooled,
   otherwise the correlation is computed per object (pooling the two contexts).

The resulting correlation reflects trial-to-trial co-fluctuations along the context axis
(noise correlation).

The residuals can be computed with sliding window time bins (BIN_SIZE wide, centers STEP_SIZE
apart, see create_time_residuals.py). The time grid around object entry then has a resolution of
STEP_SIZE, and the time of a bin refers to its center, so a bin at time t contains activity from
t - BIN_SIZE / 2 to t + BIN_SIZE / 2. Bins close to an epoch boundary therefore also contain
some activity from the neighboring epoch.

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from msvr_functions import paths, figure_style, load_objects, load_subjects

colors, dpi = figure_style()

# Settings
MIN_SPEED = 20  # mm/s, has to match create_time_residuals.py
BIN_SIZE = 0.2  # s, width of the time bins, has to match create_time_residuals.py
STEP_SIZE = 0.05  # s, spacing between bin centers, has to match create_time_residuals.py
WINDOW = [-1, 1]  # s around object entry
AXIS_WINDOWS = [[-0.8, -0.2], [0.2, 0.8]]  # s around object entry, one LDA axis is fit per epoch and
                                  # used for all time bins in that epoch (based on bin centers)
MIN_NEURONS = 10  # minimum number of neurons per region
N_FOLDS = 10
SHRINKAGE = 'auto'  # Ledoit-Wolf shrinkage of the within-class covariance
MIN_TRIALS_GROUP = 5  # minimum number of entries per object x context group to include it
MIN_TRIALS_CORR = 20  # minimum total number of entries to compute a correlation
POOL_OBJECTS = False  # pool object 1 and 2 in the correlation (after z-scoring per object)
MOUSE_GROUP = 'all'  # 'all', 'far' or 'near': only use FAR or NEAR mice
assert MOUSE_GROUP in ['all', 'far', 'near'], "MOUSE_GROUP must be 'all', 'far' or 'near'"

# Time grid relative to object entry (in units of steps)
rel_bins = np.arange(int(np.round(WINDOW[0] / STEP_SIZE)),
                     int(np.round(WINDOW[1] / STEP_SIZE)) + 1)
rel_times = np.round(rel_bins * STEP_SIZE, 3)

# Assign every time bin to an epoch: (start, end], except the first epoch which is [start, end]
# (so the bin at t = 0 belongs to the pre-entry epoch)
epoch_t = []
for i, (start, end) in enumerate(AXIS_WINDOWS):
    in_start = (rel_times >= start - 1e-9) if i == 0 else (rel_times > start + 1e-9)
    epoch_t.append(in_start & (rel_times <= end + 1e-9))
assert np.max(np.sum(epoch_t, axis=0)) <= 1, 'AXIS_WINDOWS are overlapping'

# Load in data (same file name convention as create_time_residuals.py)
path_dict = paths()
if np.isclose(STEP_SIZE, BIN_SIZE) and np.isclose(BIN_SIZE, 0.1):
    file_name = f'residuals_time_{MIN_SPEED}mms.pickle'
else:
    file_name = (f'residuals_time_{MIN_SPEED}mms_bin{int(round(BIN_SIZE * 1000))}ms'
                 f'_step{int(round(STEP_SIZE * 1000))}ms.pickle')
with open(path_dict['google_drive_data_path'] / file_name, 'rb') as handle:
    spike_dict = pickle.load(handle)

# Check that the bin settings match the ones used to create the residuals
if 'step_size' in spike_dict:
    assert all(np.isclose(s, STEP_SIZE) for s in spike_dict['step_size']), 'STEP_SIZE mismatch'
    assert all(np.isclose(b, BIN_SIZE) for b in spike_dict['bin_size']), 'BIN_SIZE mismatch'

# Equal priors so that the decision boundary lies at the midpoint between the class means
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=SHRINKAGE, priors=[0.5, 0.5])
kfold_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


def lda_projection(model, X):
    """
    Project X onto the unit-length LDA axis, centered on the midpoint between the class means.
    With equal priors the decision function is w.x - w.(mu_A + mu_B) / 2, so dividing it by |w|
    gives the distance along the discriminant axis (positive = context B).
    """
    return model.decision_function(X) / np.linalg.norm(model.coef_[0])


def axis_samples(X_all, entry_bin_idx, labels):
    """
    Get all (non-missing) time bins of a set of object entries as LDA training samples.
    entry_bin_idx: [entries x time bins] residual bin indices (-1 = missing or outside the epoch)
    """
    lab = np.repeat(labels[:, None], entry_bin_idx.shape[1], axis=1)
    valid = entry_bin_idx >= 0
    return X_all[entry_bin_idx[valid]], lab[valid]


def project_entries(model, X_all, entry_bin_idx):
    """
    Project every time bin of a set of object entries onto the LDA axis of the model.
    Returns [entries x time bins] with NaN for missing time bins.
    """
    proj = np.full(entry_bin_idx.shape, np.nan)
    valid = entry_bin_idx >= 0
    if np.any(valid):
        proj[valid] = lda_projection(model, X_all[entry_bin_idx[valid]])
    return proj


# Group recordings by session (simultaneous recordings on different probes)
sessions = pd.DataFrame({'subject': spike_dict['subject'],
                         'date': spike_dict['date']}).astype(str).drop_duplicates()

# Only keep FAR or NEAR mice
if MOUSE_GROUP != 'all':
    subjects = load_subjects()
    group_subjects = subjects.loc[subjects['Far'] == int(MOUSE_GROUP == 'far'), 'SubjectID']
    sessions = sessions[sessions['subject'].isin(group_subjects.astype(str))]
    print(f'Using {MOUSE_GROUP.upper()} mice only: {sessions.shape[0]} sessions')

decode_results, axis_results, corr_results = [], [], []
for subject, date in zip(sessions['subject'], sessions['date']):
    print(f'Processing {subject} {date}')

    # Find all recording indices for this session
    rec_indices = [idx for idx in range(len(spike_dict['date']))
                   if (str(spike_dict['date'][idx]) == date)
                   and (str(spike_dict['subject'][idx]) == subject)]

    # Only use time bins that are present in all probes of this session
    bin_keys = [np.round(spike_dict['time'][idx] / STEP_SIZE).astype(int) for idx in rec_indices]
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
        rel_bin = np.round((bin_times[trial_bins] - obj_row.times) / STEP_SIZE).astype(int)
        for b, this_rel_bin in zip(trial_bins, rel_bin):
            if this_rel_bin in rel_bins:
                bin_idx[e, this_rel_bin - rel_bins[0]] = b

    # Find all unique regions recorded in this session across all probes
    session_regions = []
    for idx in rec_indices:
        session_regions.extend(np.unique(spike_dict['region'][idx]))
    session_regions = [r for r in np.unique(session_regions) if r != 'root']

    # Cross-validated projections per region: [all object entries x time bins]
    projections = {}
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
        for obj in [1, 2]:
            obj_entries = np.where(object_labels == obj)[0]
            obj_bin_idx = bin_idx[obj_entries]
            obj_contexts = context_labels[obj_entries]

            obj_proj = np.full(obj_bin_idx.shape, np.nan)
            for epoch, (axis_window, ep_mask) in enumerate(zip(AXIS_WINDOWS, epoch_t)):

                # Only keep the time bins of this epoch
                ep_bin_idx = obj_bin_idx.copy()
                ep_bin_idx[:, ~ep_mask] = -1

                # Entries with at least one time bin in this epoch
                fit_entries = np.where(np.any(ep_bin_idx >= 0, axis=1))[0]
                y_fit = obj_contexts[fit_entries]
                if ((np.unique(y_fit).shape[0] < 2)
                        or (np.min(np.bincount(y_fit)[1:]) < N_FOLDS)):
                    continue

                # Cross-validated fixed axis: fit on the time bins in this epoch of the training
                # entries and project the time bins in this epoch of the held-out entries
                for train_index, test_index in kfold_cv.split(fit_entries, y_fit):
                    train_entries = fit_entries[train_index]
                    test_entries = fit_entries[test_index]
                    lda.fit(*axis_samples(X_all, ep_bin_idx[train_entries],
                                          obj_contexts[train_entries]))
                    test_proj = project_entries(lda, X_all, ep_bin_idx[test_entries])
                    obj_proj[np.ix_(test_entries, ep_mask)] = test_proj[:, ep_mask]

                # Axis of this epoch fit on all entries
                lda.fit(*axis_samples(X_all, ep_bin_idx[fit_entries], y_fit))
                axis_results.append({
                    'object': obj, 'epoch': epoch, 'axis_window': axis_window,
                    'region': region, 'subject': subject, 'date': date,
                    'neuron_id': neuron_ids, 'n_trials': fit_entries.shape[0],
                    'lda_axis': lda.coef_[0] / np.linalg.norm(lda.coef_[0])})
            region_proj[obj_entries] = obj_proj

            # Decoding accuracy and d' of the held-out projections per time bin
            for t, rel_time in enumerate(rel_times):
                valid = ~np.isnan(obj_proj[:, t])
                proj, y = obj_proj[valid, t], obj_contexts[valid]
                if (np.sum(y == 1) < MIN_TRIALS_GROUP) or (np.sum(y == 2) < MIN_TRIALS_GROUP):
                    continue
                pred = np.where(proj > 0, 2, 1)
                proj_A, proj_B = proj[y == 1], proj[y == 2]
                d_prime = ((np.mean(proj_B) - np.mean(proj_A))
                           / np.sqrt((np.var(proj_A) + np.var(proj_B)) / 2))
                decode_results.append({
                    'time': rel_time, 'object': obj, 'accuracy': np.mean(pred == y),
                    'd_prime': d_prime, 'n_trials': y.shape[0], 'n_neurons': X_all.shape[1],
                    'region': region, 'subject': subject, 'date': date})

        projections[region] = region_proj

    # Pooled noise correlation: z-score the projections within every object x context group,
    # concatenate the groups and correlate between region pairs
    object_sets = {'pooled': [1, 2]} if POOL_OBJECTS else {1: [1], 2: [2]}
    for region1, region2 in combinations(projections.keys(), 2):
        for t, rel_time in enumerate(rel_times):
            proj1 = projections[region1][:, t]
            proj2 = projections[region2][:, t]
            both = ~np.isnan(proj1) & ~np.isnan(proj2)
            for obj_name, objects in object_sets.items():
                z1, z2, n_groups = [], [], 0
                for obj in objects:
                    for context in [1, 2]:
                        use = both & (object_labels == obj) & (context_labels == context)
                        if np.sum(use) < MIN_TRIALS_GROUP:
                            continue
                        p1, p2 = proj1[use], proj2[use]
                        if (np.std(p1) == 0) or (np.std(p2) == 0):
                            continue
                        z1.append((p1 - np.mean(p1)) / np.std(p1))
                        z2.append((p2 - np.mean(p2)) / np.std(p2))
                        n_groups += 1
                if n_groups == 0:
                    continue
                z1, z2 = np.concatenate(z1), np.concatenate(z2)
                if z1.shape[0] < MIN_TRIALS_CORR:
                    continue

                corr_results.append({
                    'time': rel_time, 'object': obj_name, 'r': np.corrcoef(z1, z2)[0, 1],
                    'n_trials': z1.shape[0], 'n_groups': n_groups,
                    'region1': region1, 'region2': region2,
                    'region_pair': f'{region1}-{region2}', 'subject': subject, 'date': date})

decode_df = pd.DataFrame(decode_results)
axis_df = pd.DataFrame(axis_results)
corr_df = pd.DataFrame(corr_results)

# Save to disk (file names include the bin settings)
suffix = f'_bin{int(round(BIN_SIZE * 1000))}ms_step{int(round(STEP_SIZE * 1000))}ms'
if MOUSE_GROUP != 'all':
    suffix += f'_{MOUSE_GROUP}'
decode_df.to_csv(path_dict['google_drive_data_path'] / f'decode_object_residuals_lda_fixed_axis{suffix}.csv', index=False)
corr_df.to_csv(path_dict['google_drive_data_path'] / f'region_corr_lda_fixed_axis{suffix}.csv', index=False)

# %% Plot LDA decoding example
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
plot_df = decode_df[(decode_df['region'] == 'TEa') & (decode_df['object'] == 2)]
sns.lineplot(data=plot_df, x='time', y='accuracy', ax=ax1, color='k',
                errorbar='se', err_kws={'lw': 0})
ax1.set(ylabel='LDA decoding accuracy (%)', xlabel='Time from object entry (s)',
        yticks=[0.5, 0.6, 0.7, 0.8], yticklabels=[50, 60, 70, 80], ylim=[0.48, 0.8])
ax1.axvline(0, ls=':', lw=0.5, color='grey', zorder=0)
ax1.axhline(0.5, ls=':', lw=0.5, color='grey', zorder=0)

sns.despine(trim=True)
plt.tight_layout()

# %% Plot LDA correlation example
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
plot_df = decode_df[(decode_df['region'] == 'TEa') & (decode_df['object'] == 2)]
sns.lineplot(data=plot_df, x='time', y='accuracy', ax=ax1, color='k',
                errorbar='se', err_kws={'lw': 0})
ax1.set(ylabel='LDA decoding accuracy (%)', xlabel='Time from object entry (s)',
        yticks=[0.5, 0.6, 0.7, 0.8], yticklabels=[50, 60, 70, 80], ylim=[0.48, 0.8])
ax1.axvline(0, ls=':', lw=0.5, color='grey', zorder=0)
ax1.axhline(0.5, ls=':', lw=0.5, color='grey', zorder=0)

sns.despine(trim=True)
plt.tight_layout()

# %% Plot LDA decoding accuracy on the fixed axis per region, separately per object
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
plot_df = decode_df[(decode_df['region'] == 'TEa') & (decode_df['object'] == 2)]
sns.lineplot(data=plot_df, x='time', y='accuracy', ax=ax1, color='k',
                errorbar='se', err_kws={'lw': 0})
ax1.set(ylabel='LDA decoding accuracy (%)', xlabel='Time from object entry (s)',
        yticks=[0.5, 0.6, 0.7, 0.8], yticklabels=[50, 60, 70, 80], ylim=[0.48, 0.8])
ax1.axvline(0, ls=':', lw=0.5, color='grey', zorder=0)
ax1.axhline(0.5, ls=':', lw=0.5, color='grey', zorder=0)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'decode_TEa.pdf')

f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
plot_df = corr_df[(corr_df['region_pair'] == 'CA1-TEa') & (corr_df['object'] == 2)]
sns.lineplot(data=plot_df, x='time', y='r', ax=ax1, color='k',
                errorbar='se', err_kws={'lw': 0})
ax1.set(ylabel='LDA projection correlation (r)', xlabel='Time from object entry (s)')
ax1.axvline(0, ls=':', lw=0.5, color='grey', zorder=0)
ax1.axhline(0, ls=':', lw=0.5, color='grey', zorder=0)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'correlate_CA1_TEa.pdf')

# %%

# Boundaries between epochs (other than object entry, which is always drawn)
epoch_boundaries = sorted({w[0] for w in AXIS_WINDOWS[1:]} - {0})

regions = sorted(decode_df['region'].unique())
n_cols = 4
n_rows = int(np.ceil(len(regions) / n_cols))
for obj in [1, 2]:
    f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.75, n_rows * 1.75), dpi=dpi,
                          sharex=True, squeeze=False)
    axs = axs.flatten()
    for r, region in enumerate(regions):
        ax = axs[r]
        plot_df = decode_df[(decode_df['region'] == region) & (decode_df['object'] == obj)]
        for boundary in epoch_boundaries:
            ax.axvline(boundary, ls=':', lw=0.5, color='grey', zorder=0)
        sns.lineplot(data=plot_df, x='time', y='accuracy', ax=ax, color='k',
                     errorbar='se', err_kws={'lw': 0})
        ax.axhline(0.5, ls='--', lw=0.5, color='grey', zorder=0)
        ax.axvline(0, ls='--', lw=0.5, color='k', zorder=0)
        n_rec = plot_df[['subject', 'date']].drop_duplicates().shape[0]
        ax.set(title=f'{region} (n={n_rec})', xlabel='', ylabel='')
        if r % n_cols == 0:
            ax.set_ylabel('LDA context accuracy')
    for ax in axs[len(regions):]:
        ax.axis('off')
    f.suptitle(f'Object {obj}', fontsize=7)
    f.supxlabel('Time from object entry (s)', fontsize=7)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'decode_object{obj}_residuals_lda_fixed_axis_accuracy{suffix}.pdf')
    plt.show()

# %% Plot pooled noise correlation of the fixed-axis LDA projections between region pairs

region_pairs = sorted(corr_df['region_pair'].unique())
n_cols = 4
n_rows = int(np.ceil(len(region_pairs) / n_cols))
for obj_name in corr_df['object'].unique():
    f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.75, n_rows * 1.75), dpi=dpi,
                          sharex=True, squeeze=False)
    axs = axs.flatten()
    for p, region_pair in enumerate(region_pairs):
        ax = axs[p]
        plot_df = corr_df[(corr_df['region_pair'] == region_pair)
                          & (corr_df['object'] == obj_name)]
        for boundary in epoch_boundaries:
            ax.axvline(boundary, ls=':', lw=0.5, color='grey', zorder=0)
        sns.lineplot(data=plot_df, x='time', y='r', ax=ax, color='k',
                     errorbar='se', err_kws={'lw': 0})
        ax.axhline(0, ls='--', lw=0.5, color='grey', zorder=0)
        ax.axvline(0, ls='--', lw=0.5, color='k', zorder=0)
        n_rec = plot_df[['subject', 'date']].drop_duplicates().shape[0]
        ax.set(title=f'{region_pair} (n={n_rec})', xlabel='', ylabel='')
        if p % n_cols == 0:
            ax.set_ylabel('LDA projection correlation (r)')
    for ax in axs[len(region_pairs):]:
        ax.axis('off')
    title = 'Objects pooled' if obj_name == 'pooled' else f'Object {obj_name}'
    f.suptitle(title, fontsize=7)
    f.supxlabel('Time from object entry (s)', fontsize=7)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'region_corr_lda_fixed_axis_object-{obj_name}{suffix}.pdf')
    plt.show()
