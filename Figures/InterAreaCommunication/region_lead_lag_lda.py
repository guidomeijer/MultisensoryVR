# %% -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 2026

Lead-lag relationship between two brain regions around a chosen time bin, for one object.

Uses the same fixed-axis LDA projections as region_corr_lda.py, but only for one region pair
and one object, and correlates the projections of the two regions at different time bins:

1) Fixed LDA axis: per session and region, a single LDA axis that separates context A from
   context B is fit on the time-binned GLM residuals (output of create_time_residuals.py) of all
   time bins in the analysis window (CENTER_TIME +- MAX_LAG). The axis is cross-validated over
   object entries (N_FOLDS-fold): held-out entries are projected onto the axis fit on the other
   entries. Because one axis is used for the whole window, differences between lags reflect the
   data and not changes in the axis.

2) Lagged noise correlation: for every combination of a REGION1 time bin t1 and a REGION2 time
   bin t2 in the window, the projections are z-scored within each context (removing the context
   signal) and correlated over object entries, giving a [t1 x t2] correlation matrix per session.

3) Centered cross-correlogram: for lag tau = t2 - t1, the correlation is taken from the pair of
   time bins that is centered on CENTER_TIME (t1 = CENTER_TIME - tau / 2, t2 = CENTER_TIME +
   tau / 2). For lags of an odd number of steps there is no exactly centered pair, so the two
   pairs centered half a step before and after CENTER_TIME are averaged. Both directions
   therefore span the same stretch of time, so a time-varying correlation strength does not
   bias the comparison between positive and negative lags.

   Positive lag = REGION2 later than REGION1. A correlogram that is larger at positive than at
   negative lags means that REGION1 activity predicts later REGION2 activity (REGION1 leads).
   The asymmetry index is mean r(tau > 0) - mean r(tau < 0) per session (positive = REGION1
   leads), which is tested against zero over sessions.

Note: with sliding window bins (BIN_SIZE > STEP_SIZE) neighboring bins overlap, which smooths the
correlogram over roughly BIN_SIZE. The asymmetry is still interpretable, but lags shorter than
BIN_SIZE are blurred. Use BIN_SIZE = STEP_SIZE = 0.1 for non-overlapping bins.

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from msvr_functions import paths, figure_style, load_objects, load_subjects

colors, dpi = figure_style()

# Analysis selection
REGION1 = 'CA1'
REGION2 = 'TEa'
CENTER_TIME = -0.5  # s relative to object entry, time bin on which the analysis is centered
OBJECT = 2  # 1 or 2
MAX_LAG = 0.4  # s, maximum lag between the two regions

# Settings
MIN_SPEED = 20  # mm/s, has to match create_time_residuals.py
BIN_SIZE = 0.1  # s, width of the time bins, has to match create_time_residuals.py
STEP_SIZE = 0.1  # s, spacing between bin centers, has to match create_time_residuals.py
MIN_NEURONS = 10  # minimum number of neurons per region
N_FOLDS = 10
SHRINKAGE = 'auto'  # Ledoit-Wolf shrinkage of the within-class covariance
MIN_TRIALS_GROUP = 5  # minimum number of entries per context to include it in the correlation
MIN_TRIALS_CORR = 20  # minimum total number of entries to compute a correlation
MOUSE_GROUP = 'all'  # 'all', 'far' or 'near': only use FAR or NEAR mice
assert MOUSE_GROUP in ['all', 'far', 'near'], "MOUSE_GROUP must be 'all', 'far' or 'near'"
assert OBJECT in [1, 2], 'OBJECT must be 1 or 2'
assert REGION1 != REGION2, 'Pick two different regions'

# Time grid of the analysis window relative to object entry (in units of steps)
max_lag_steps = int(np.round(MAX_LAG / STEP_SIZE))
center_step = int(np.round(CENTER_TIME / STEP_SIZE))
rel_bins = np.arange(center_step - max_lag_steps, center_step + max_lag_steps + 1)
rel_times = np.round(rel_bins * STEP_SIZE, 3)
c_idx = max_lag_steps  # index of the center bin in the window
lag_steps = np.arange(-max_lag_steps, max_lag_steps + 1)
lag_times = np.round(lag_steps * STEP_SIZE, 3)

# Load in data (same file name convention as create_time_residuals.py)
path_dict = paths()
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
    Project X onto the unit-length LDA axis, centered on the midpoint between the class means
    (positive = context B).
    """
    return model.decision_function(X) / np.linalg.norm(model.coef_[0])


def axis_samples(X_all, entry_bin_idx, labels):
    """
    Get all (non-missing) time bins of a set of object entries as LDA training samples.
    entry_bin_idx: [entries x time bins] residual bin indices (-1 = missing)
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


def noise_corr(p1, p2, contexts):
    """
    Correlate two projections over entries after z-scoring within each context.
    Returns NaN if there are not enough entries.
    """
    both = ~np.isnan(p1) & ~np.isnan(p2)
    z1, z2 = [], []
    for context in [1, 2]:
        use = both & (contexts == context)
        if np.sum(use) < MIN_TRIALS_GROUP:
            continue
        if (np.std(p1[use]) == 0) or (np.std(p2[use]) == 0):
            continue
        z1.append((p1[use] - np.mean(p1[use])) / np.std(p1[use]))
        z2.append((p2[use] - np.mean(p2[use])) / np.std(p2[use]))
    if len(z1) == 0:
        return np.nan
    z1, z2 = np.concatenate(z1), np.concatenate(z2)
    if z1.shape[0] < MIN_TRIALS_CORR:
        return np.nan
    return np.corrcoef(z1, z2)[0, 1]


def centered_correlogram(corr_mat):
    """
    Cross-correlogram centered on the center bin: for lag k (steps, region 2 relative to
    region 1) take corr_mat[i, i + k] with (i + (i + k)) / 2 = center. For odd lags the two pairs
    centered half a step before and after the center are averaged.
    """
    ccg = np.full(lag_steps.shape[0], np.nan)
    for l, k in enumerate(lag_steps):
        if k % 2 == 0:
            ccg[l] = corr_mat[c_idx - k // 2, c_idx + k // 2]
        else:
            vals = []
            for pair_sum in [2 * c_idx - 1, 2 * c_idx + 1]:
                i = (pair_sum - k) // 2
                j = i + k
                if (0 <= i < corr_mat.shape[0]) and (0 <= j < corr_mat.shape[1]):
                    vals.append(corr_mat[i, j])
            if len(vals) > 0 and not np.all(np.isnan(vals)):
                ccg[l] = np.nanmean(vals)
    return ccg


def region_data(rec_indices, bin_masks, region):
    """ Z-scored residuals [time bins x neurons] of one region, pooled over probes """
    region_res = []
    for idx, bin_mask in zip(rec_indices, bin_masks):
        mask = spike_dict['region'][idx] == region
        if np.sum(mask) > 0:
            region_res.append(spike_dict['residuals'][idx][bin_mask][:, mask])
    if len(region_res) == 0:
        return None
    X_all = np.hstack(region_res)
    X_all = X_all[:, np.std(X_all, axis=0) > 0]
    if X_all.shape[1] < MIN_NEURONS:
        return None
    return (X_all - np.mean(X_all, axis=0)) / np.std(X_all, axis=0)


# Group recordings by session (simultaneous recordings on different probes)
sessions = pd.DataFrame({'subject': spike_dict['subject'],
                         'date': spike_dict['date']}).astype(str).drop_duplicates()

# Only keep FAR or NEAR mice
if MOUSE_GROUP != 'all':
    subjects = load_subjects()
    group_subjects = subjects.loc[subjects['Far'] == int(MOUSE_GROUP == 'far'), 'SubjectID']
    sessions = sessions[sessions['subject'].isin(group_subjects.astype(str))]
    print(f'Using {MOUSE_GROUP.upper()} mice only: {sessions.shape[0]} sessions')

corr_mats, ccg_results = [], []
for subject, date in zip(sessions['subject'], sessions['date']):

    # Find all recording indices for this session
    rec_indices = [idx for idx in range(len(spike_dict['date']))
                   if (str(spike_dict['date'][idx]) == date)
                   and (str(spike_dict['subject'][idx]) == subject)]

    # Skip sessions in which not both regions were recorded
    session_regions = np.unique(np.concatenate([spike_dict['region'][idx]
                                                for idx in rec_indices]))
    if (REGION1 not in session_regions) or (REGION2 not in session_regions):
        continue

    # Only use time bins that are present in all probes of this session
    bin_keys = [np.round(spike_dict['time'][idx] / STEP_SIZE).astype(int) for idx in rec_indices]
    shared_keys = bin_keys[0]
    for keys in bin_keys[1:]:
        shared_keys = np.intersect1d(shared_keys, keys)
    bin_masks = [np.isin(keys, shared_keys) for keys in bin_keys]
    ref_idx, ref_mask = rec_indices[0], bin_masks[0]
    bin_times = spike_dict['time'][ref_idx][ref_mask]
    trial_per_bin = spike_dict['trial'][ref_idx][ref_mask]

    # Get residuals of both regions
    X1 = region_data(rec_indices, bin_masks, REGION1)
    X2 = region_data(rec_indices, bin_masks, REGION2)
    if (X1 is None) or (X2 is None):
        continue
    print(f'Processing {subject} {date} ({REGION1}: {X1.shape[1]}, {REGION2}: {X2.shape[1]} neurons)')

    # Load in entries of the selected object
    obj_df = load_objects(subject, date)
    obj_df = obj_df[(obj_df['object'] == OBJECT) & ~np.isnan(obj_df['times'])]
    obj_df = obj_df.reset_index(drop=True)
    contexts = obj_df['sound'].values

    # For every object entry and relative time, find the corresponding residual time bin
    # (-1 if this bin is missing, e.g. because the mouse was running too slow)
    bin_idx = np.full((obj_df.shape[0], rel_bins.shape[0]), -1)
    for e, obj_row in enumerate(obj_df.itertuples()):
        trial_bins = np.where(trial_per_bin == obj_row.trial_nr)[0]
        rel_bin = np.round((bin_times[trial_bins] - obj_row.times) / STEP_SIZE).astype(int)
        for b, this_rel_bin in zip(trial_bins, rel_bin):
            if this_rel_bin in rel_bins:
                bin_idx[e, this_rel_bin - rel_bins[0]] = b

    # Entries with at least one time bin in the window
    fit_entries = np.where(np.any(bin_idx >= 0, axis=1))[0]
    y_fit = contexts[fit_entries]
    if (np.unique(y_fit).shape[0] < 2) or (np.min(np.bincount(y_fit)[1:]) < N_FOLDS):
        continue

    # Cross-validated projections onto one fixed LDA axis per region: [entries x time bins]
    projections = []
    for X_all in [X1, X2]:
        proj = np.full(bin_idx.shape, np.nan)
        for train_index, test_index in kfold_cv.split(fit_entries, y_fit):
            train_entries, test_entries = fit_entries[train_index], fit_entries[test_index]
            lda.fit(*axis_samples(X_all, bin_idx[train_entries], contexts[train_entries]))
            proj[test_entries] = project_entries(lda, X_all, bin_idx[test_entries])
        projections.append(proj)

    # Lagged noise correlation matrix [REGION1 time x REGION2 time]
    corr_mat = np.full((rel_bins.shape[0], rel_bins.shape[0]), np.nan)
    for i in range(rel_bins.shape[0]):
        for j in range(rel_bins.shape[0]):
            corr_mat[i, j] = noise_corr(projections[0][:, i], projections[1][:, j], contexts)
    if np.all(np.isnan(corr_mat)):
        continue
    corr_mats.append(corr_mat)

    # Centered cross-correlogram and asymmetry index
    ccg = centered_correlogram(corr_mat)
    asymmetry = np.nanmean(ccg[lag_steps > 0]) - np.nanmean(ccg[lag_steps < 0])
    for lag, r in zip(lag_times, ccg):
        ccg_results.append({'lag': lag, 'r': r, 'asymmetry': asymmetry,
                            'n_trials': fit_entries.shape[0], 'n_neurons1': X1.shape[1],
                            'n_neurons2': X2.shape[1], 'subject': subject, 'date': date})

ccg_df = pd.DataFrame(ccg_results)
corr_mats = np.array(corr_mats)
n_sessions = corr_mats.shape[0]
print(f'\n{REGION1}-{REGION2}, object {OBJECT}, centered on {CENTER_TIME} s: {n_sessions} sessions')

# Statistics on the asymmetry index over sessions
asym_df = ccg_df.groupby(['subject', 'date'])['asymmetry'].first().reset_index()
_, p_asym = wilcoxon(asym_df['asymmetry'])
mean_ccg = ccg_df.groupby('lag')['r'].mean()
print(f'Asymmetry index: {asym_df["asymmetry"].mean():.3f} +- {asym_df["asymmetry"].sem():.3f} '
      f'(Wilcoxon p = {p_asym:.3g}), positive = {REGION1} leads')
print(f'Peak of the mean correlogram at {mean_ccg.idxmax()} s lag')

# %% Plot the lagged correlation matrix
fig_name = f'lead_lag_{REGION1}_{REGION2}_object{OBJECT}_t{CENTER_TIME}'
if MOUSE_GROUP != 'all':
    fig_name += f'_{MOUSE_GROUP}'

f, ax = plt.subplots(figsize=(2.2, 1.9), dpi=dpi)
mean_mat = np.nanmean(corr_mats, axis=0)
v_max = np.nanmax(np.abs(mean_mat))
half_step = STEP_SIZE / 2
extent = [rel_times[0] - half_step, rel_times[-1] + half_step,
          rel_times[0] - half_step, rel_times[-1] + half_step]
img = ax.imshow(mean_mat, origin='lower', extent=extent, cmap='coolwarm',
                vmin=-v_max, vmax=v_max, aspect='auto')
ax.plot(rel_times, rel_times, ls='--', lw=0.5, color='k')  # zero lag
ax.plot(rel_times, 2 * CENTER_TIME - rel_times, ls=':', lw=0.5, color='k')  # centered pairs
ax.set(xlabel=f'{REGION2} time (s)', ylabel=f'{REGION1} time (s)',
       title=f'Object {OBJECT} (n={n_sessions})')
cbar = f.colorbar(img, ax=ax, shrink=0.8)
cbar.set_label('Noise correlation (r)')
plt.tight_layout()
# plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'{fig_name}_matrix.pdf')
plt.show()

# %% Plot the centered cross-correlogram and the asymmetry index
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.2, 1.75), dpi=dpi,
                             gridspec_kw={'width_ratios': [2.5, 1]})
sns.lineplot(data=ccg_df, x='lag', y='r', ax=ax1, color='k', errorbar='se',
             err_kws={'lw': 0})
ax1.axvline(0, ls='--', lw=0.5, color='grey', zorder=0)
ax1.axhline(0, ls='--', lw=0.5, color='grey', zorder=0)
ax1.set(xlabel=f'Lag (s)\n← {REGION2} leads   {REGION1} leads →',
        ylabel='Noise correlation (r)', title=f'Centered on {CENTER_TIME} s')

sns.stripplot(y=asym_df['asymmetry'], ax=ax2, color='grey', size=2.5, jitter=0.15, zorder=1)
ax2.errorbar(0, asym_df['asymmetry'].mean(), yerr=asym_df['asymmetry'].sem(), fmt='o',
             color='k', ms=3, capsize=0, zorder=2)
ax2.axhline(0, ls='--', lw=0.5, color='grey', zorder=0)
ax2.set(xticks=[], ylabel='Asymmetry index', title=f'p = {p_asym:.2g}')

sns.despine(trim=True)
sns.despine(ax=ax2, bottom=True)
plt.tight_layout()
# plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'{fig_name}_ccg.pdf')
plt.show()
