# %% -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 2026

Plot the noise correlations of the fixed-axis LDA projections between brain regions as a network,
with significance determined by a trial-shuffle control.

1) The cross-validated fixed-axis LDA projections are computed exactly as in region_corr_lda.py
   (one context axis per session, region, object and epoch in AXIS_WINDOWS).

2) Per session, region pair, object and time bin, the pooled noise correlation is computed as in
   region_corr_lda.py: the projections are z-scored within every context group and the groups are
   concatenated and correlated. The correlation is then averaged over the time bins of each
   summary epoch (EPOCHS), giving the mean correlation (= AUC / epoch duration), which does not
   depend on the epoch duration or on missing time bins.

3) Shuffle control: the object entries of region2 are randomly permuted within every context
   group (N_SHUFFLES times). The same permutation is used for all time bins of an epoch, so the
   temporal structure within an entry is kept and only the trial-to-trial pairing between the two
   regions is destroyed. The shuffled correlations go through exactly the same pipeline (z-scoring,
   pooling, epoch averaging) as the observed ones.

4) Group statistics: per region pair, object and epoch, the observed mean correlation averaged over
   sessions is compared to the null distribution of the session-averaged shuffled correlations
   (two-sided permutation p-value). The p-values are corrected for multiple comparisons over
   region pairs with the Benjamini-Hochberg FDR, separately per object x epoch combination.

By Guido Meijer
"""

import warnings
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from itertools import combinations
from scipy.stats import false_discovery_control
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from msvr_functions import paths, figure_style, load_objects, load_subjects
colors, dpi = figure_style()

# Settings that have to match region_corr_lda.py
MIN_SPEED = 20  # mm/s, has to match create_time_residuals.py
BIN_SIZE = 0.2  # s, has to match create_time_residuals.py
STEP_SIZE = 0.05  # s, has to match create_time_residuals.py
WINDOW = [-1, 1]  # s around object entry
AXIS_WINDOWS = [[-1, 0], [0, 1]]  # s around object entry, one LDA axis per epoch
MIN_NEURONS = 10
N_FOLDS = 10
SHRINKAGE = 'auto'
MIN_TRIALS_GROUP = 5  # minimum number of entries per context group to include it
MIN_TRIALS_CORR = 20  # minimum total number of entries to compute a correlation
MOUSE_GROUP = 'all'  # 'all', 'far' or 'near'
assert MOUSE_GROUP in ['all', 'far', 'near'], "MOUSE_GROUP must be 'all', 'far' or 'near'"

# Settings of the network analysis
EPOCHS = {'Pre-entry (-0.5 to 0 s)': [-0.5, 0],
          'Post-entry (0.2 to 1 s)': [0.2, 1]}
MIN_COVERAGE = 0.8  # minimum fraction of time bins in an epoch that need a correlation value
MIN_SESSIONS = 3  # minimum number of sessions for a region pair to be plotted
N_SHUFFLES = 1000
ONLY_SIGNIFICANT = True  # only draw region pairs that differ significantly from the shuffles
ALPHA = 0.05
NODE_ORDER = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']  # regions not in this list are appended
CMAP = 'coolwarm'
rng = np.random.default_rng(42)

# Time grid relative to object entry (in units of steps)
rel_bins = np.arange(int(np.round(WINDOW[0] / STEP_SIZE)),
                     int(np.round(WINDOW[1] / STEP_SIZE)) + 1)
rel_times = np.round(rel_bins * STEP_SIZE, 3)


def epoch_masks_from_windows(windows):
    """
    Assign every time bin to an epoch based on its center: (start, end], except the first epoch
    which is [start, end] (so the bin at t = 0 belongs to the pre-entry epoch).
    """
    masks = []
    for i, (start, end) in enumerate(windows):
        in_start = (rel_times >= start - 1e-9) if i == 0 else (rel_times > start + 1e-9)
        masks.append(in_start & (rel_times <= end + 1e-9))
    return masks


axis_epoch_t = epoch_masks_from_windows(AXIS_WINDOWS)
assert np.max(np.sum(axis_epoch_t, axis=0)) <= 1, 'AXIS_WINDOWS are overlapping'
summary_epoch_t = dict(zip(EPOCHS.keys(), epoch_masks_from_windows(list(EPOCHS.values()))))
corr_t = np.where(np.any(list(summary_epoch_t.values()), axis=0))[0]  # bins to correlate

# Load in data (same file name convention as create_time_residuals.py)
path_dict = paths()
if np.isclose(STEP_SIZE, BIN_SIZE) and np.isclose(BIN_SIZE, 0.1):
    file_name = f'residuals_time_{MIN_SPEED}mms.pickle'
else:
    file_name = (f'residuals_time_{MIN_SPEED}mms_bin{int(round(BIN_SIZE * 1000))}ms'
                 f'_step{int(round(STEP_SIZE * 1000))}ms.pickle')
with open(path_dict['google_drive_data_path'] / file_name, 'rb') as handle:
    spike_dict = pickle.load(handle)
if 'step_size' in spike_dict:
    assert all(np.isclose(s, STEP_SIZE) for s in spike_dict['step_size']), 'STEP_SIZE mismatch'
    assert all(np.isclose(b, BIN_SIZE) for b in spike_dict['bin_size']), 'BIN_SIZE mismatch'

# Equal priors so that the decision boundary lies at the midpoint between the class means
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=SHRINKAGE, priors=[0.5, 0.5])
kfold_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


def lda_projection(model, X):
    """Project X onto the unit-length LDA axis, centered on the midpoint between class means."""
    return model.decision_function(X) / np.linalg.norm(model.coef_[0])


def axis_samples(X_all, entry_bin_idx, labels):
    """Get all (non-missing) time bins of a set of object entries as LDA training samples."""
    lab = np.repeat(labels[:, None], entry_bin_idx.shape[1], axis=1)
    valid = entry_bin_idx >= 0
    return X_all[entry_bin_idx[valid]], lab[valid]


def project_entries(model, X_all, entry_bin_idx):
    """Project every time bin of a set of object entries onto the LDA axis (NaN if missing)."""
    proj = np.full(entry_bin_idx.shape, np.nan)
    valid = entry_bin_idx >= 0
    if np.any(valid):
        proj[valid] = lda_projection(model, X_all[entry_bin_idx[valid]])
    return proj


def pooled_corr(proj1, proj2, group_masks):
    """
    Pooled noise correlation for many (shuffled) versions of proj2 at once.
    proj1: [entries], proj2: [versions x entries], group_masks: boolean [entries] per context.
    Per version, both projections are z-scored within every group (using the entries that have a
    value in both regions) and the groups are concatenated and correlated. Because every group is
    z-scored (mean 0, population SD 1), the pooled vectors have mean 0 and SD 1, so the Pearson
    correlation equals the mean of the product. Identical to np.corrcoef in region_corr_lda.py.
    Returns the correlation per version [versions] (NaN if not enough entries).
    """
    n_versions = proj2.shape[0]
    prod_sum, n_total = np.zeros(n_versions), np.zeros(n_versions)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for g in group_masks:
            p2 = proj2[:, g]
            valid = ~np.isnan(proj1[g])[None, :] & ~np.isnan(p2)
            n = np.sum(valid, axis=1)
            p1 = np.where(valid, proj1[g][None, :], np.nan)
            p2 = np.where(valid, p2, np.nan)
            sd1, sd2 = np.nanstd(p1, axis=1), np.nanstd(p2, axis=1)
            use = (n >= MIN_TRIALS_GROUP) & (sd1 > 0) & (sd2 > 0)
            z1 = (p1 - np.nanmean(p1, axis=1, keepdims=True)) / sd1[:, None]
            z2 = (p2 - np.nanmean(p2, axis=1, keepdims=True)) / sd2[:, None]
            prod_sum[use] += np.nansum(z1 * z2, axis=1)[use]
            n_total[use] += n[use]
        r = prod_sum / n_total
    r[n_total < MIN_TRIALS_CORR] = np.nan
    return r


# Group recordings by session (simultaneous recordings on different probes)
sessions = pd.DataFrame({'subject': spike_dict['subject'],
                         'date': spike_dict['date']}).astype(str).drop_duplicates()
if MOUSE_GROUP != 'all':
    subjects = load_subjects()
    group_subjects = subjects.loc[subjects['Far'] == int(MOUSE_GROUP == 'far'), 'SubjectID']
    sessions = sessions[sessions['subject'].isin(group_subjects.astype(str))]
    print(f'Using {MOUSE_GROUP.upper()} mice only: {sessions.shape[0]} sessions')

epoch_results = []
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
        region_res = []
        for idx, bin_mask in zip(rec_indices, bin_masks):
            mask = spike_dict['region'][idx] == region
            if np.sum(mask) > 0:
                region_res.append(spike_dict['residuals'][idx][bin_mask][:, mask])
        X_all = np.hstack(region_res)
        X_all = X_all[:, np.std(X_all, axis=0) > 0]
        if X_all.shape[1] < MIN_NEURONS:
            continue
        X_all = (X_all - np.mean(X_all, axis=0)) / np.std(X_all, axis=0)

        region_proj = np.full(bin_idx.shape, np.nan)
        for obj in [1, 2]:
            obj_entries = np.where(object_labels == obj)[0]
            obj_bin_idx = bin_idx[obj_entries]
            obj_contexts = context_labels[obj_entries]
            obj_proj = np.full(obj_bin_idx.shape, np.nan)
            for ep_mask in axis_epoch_t:
                ep_bin_idx = obj_bin_idx.copy()
                ep_bin_idx[:, ~ep_mask] = -1
                fit_entries = np.where(np.any(ep_bin_idx >= 0, axis=1))[0]
                y_fit = obj_contexts[fit_entries]
                if ((np.unique(y_fit).shape[0] < 2)
                        or (np.min(np.bincount(y_fit)[1:]) < N_FOLDS)):
                    continue
                for train_index, test_index in kfold_cv.split(fit_entries, y_fit):
                    train_entries = fit_entries[train_index]
                    test_entries = fit_entries[test_index]
                    lda.fit(*axis_samples(X_all, ep_bin_idx[train_entries],
                                          obj_contexts[train_entries]))
                    test_proj = project_entries(lda, X_all, ep_bin_idx[test_entries])
                    obj_proj[np.ix_(test_entries, ep_mask)] = test_proj[:, ep_mask]
            region_proj[obj_entries] = obj_proj
        projections[region] = region_proj

    # Observed and shuffled epoch-mean correlations per region pair and object
    for obj in [1, 2]:
        obj_entries = np.where(object_labels == obj)[0]
        obj_contexts = context_labels[obj_entries]
        group_masks = [obj_contexts == context for context in [1, 2]]

        # Entry order of region2: row 0 is the original order, the other rows are permuted
        # within every context group (the same permutation is used for all time bins and pairs)
        perm = np.tile(np.arange(obj_entries.shape[0]), (N_SHUFFLES + 1, 1))
        for g in group_masks:
            idx = np.where(g)[0]
            perm[1:, idx] = idx[rng.permuted(np.tile(np.arange(idx.shape[0]), (N_SHUFFLES, 1)),
                                             axis=1)]

        for region1, region2 in combinations(projections.keys(), 2):
            proj1_obj = projections[region1][obj_entries]
            proj2_obj = projections[region2][obj_entries]
            r_time = np.full((N_SHUFFLES + 1, rel_times.shape[0]), np.nan)
            for t in corr_t:
                r_time[:, t] = pooled_corr(proj1_obj[:, t], proj2_obj[:, t][perm], group_masks)

            # Mean correlation over the time bins of each epoch
            for epoch_name, ep_mask in summary_epoch_t.items():
                r_epoch = r_time[:, ep_mask]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    mean_r = np.nanmean(r_epoch, axis=1)
                mean_r[np.sum(~np.isnan(r_epoch), axis=1) < MIN_COVERAGE * np.sum(ep_mask)] = np.nan
                if np.isnan(mean_r[0]):
                    continue
                epoch_results.append({
                    'subject': subject, 'date': date, 'region1': region1, 'region2': region2,
                    'object': obj, 'epoch': epoch_name,
                    'r': mean_r[0], 'r_shuffle': mean_r[1:]})
epoch_df = pd.DataFrame(epoch_results)

# %% Group statistics: observed session average vs. null distribution of shuffled session averages

summary_results = []
for (region1, region2, obj, epoch_name), this_df in epoch_df.groupby(
        ['region1', 'region2', 'object', 'epoch']):
    if this_df.shape[0] < MIN_SESSIONS:
        continue
    obs = this_df['r'].mean()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        null = np.nanmean(np.vstack(this_df['r_shuffle'].values), axis=0)
    null = null[~np.isnan(null)]
    center = np.mean(null)
    p_value = ((1 + np.sum(np.abs(null - center) >= np.abs(obs - center)))
               / (null.shape[0] + 1))
    summary_results.append({
        'region1': region1, 'region2': region2, 'object': obj, 'epoch': epoch_name,
        'r': obs, 'r_shuffle': center, 'n_sessions': this_df.shape[0], 'p_value': p_value})
summary_df = pd.DataFrame(summary_results)

# Benjamini-Hochberg FDR correction over region pairs, per object x epoch combination
summary_df['p_fdr'] = np.nan
for _, this_df in summary_df.groupby(['object', 'epoch']):
    summary_df.loc[this_df.index, 'p_fdr'] = false_discovery_control(
        this_df['p_value'], method='bh')
print(summary_df.sort_values(['object', 'epoch', 'p_fdr']).to_string(index=False))

# Node order on the circle
all_regions = np.unique(np.concatenate([summary_df['region1'], summary_df['region2']]))
node_order = ([r for r in NODE_ORDER if r in all_regions]
              + [r for r in all_regions if r not in NODE_ORDER])
pos = nx.circular_layout(dict(zip(node_order, range(len(node_order)))))

# Symmetric color scale and edge width scaling shared across all panels
max_r = summary_df['r'].abs().max()
norm = mpl.colors.Normalize(vmin=-max_r, vmax=max_r)
cmap = plt.get_cmap(CMAP)

# %% Plot network per epoch, one figure per object (same color scale for all figures)

for obj in sorted(summary_df['object'].unique()):
    f, axs = plt.subplots(1, len(EPOCHS), figsize=(4.8, 2.5), dpi=dpi, squeeze=False)
    axs = axs[0]
    for j, epoch_name in enumerate(EPOCHS.keys()):
        ax = axs[j]
        df = summary_df[(summary_df['object'] == obj) & (summary_df['epoch'] == epoch_name)]
        if ONLY_SIGNIFICANT:
            df = df[df['p_fdr'] < ALPHA]

        # Build an undirected graph (correlations are symmetric)
        G = nx.Graph()
        G.add_nodes_from(node_order)
        for row in df.itertuples():
            G.add_edge(row.region1, row.region2, weight=row.r)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color='lightgrey')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

        # Draw edges: width scales with |r|, color with the sign and size of r
        edges = list(G.edges(data=True))
        if len(edges) > 0:
            weights = np.array([d['weight'] for (_, _, d) in edges])
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(u, v) for (u, v, _) in edges],
                width=0.5 + 3 * np.abs(weights) / max_r,
                edge_color=[cmap(norm(w)) for w in weights],
                node_size=500)

        x_values, y_values = zip(*pos.values())
        ax.set(xlim=(min(x_values) - 0.25, max(x_values) + 0.25),
               ylim=(min(y_values) - 0.25, max(y_values) + 0.25))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(epoch_name, fontsize=7)

    # Colorbar
    f.subplots_adjust(right=0.85)
    cax = f.add_axes([0.88, 0.25, 0.01, 0.5])
    cbar = f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label('Mean correlation (r)', fontsize=7, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=7)
    f.suptitle(f'Object {obj}', fontsize=7)

    # plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'region_corr_lda_network_shuffle_object{obj}.pdf')
    plt.show()
