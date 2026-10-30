# -*- coding: utf-8 -*-
"""
Tests whether the spatial and contextual codes occupy orthogonal subspaces in
neuron firing space by explicitly separating variance driven by position from
variance driven by context, extracting their respective basis vectors via PCA,
and measuring the principal (canonical) angles between the two subspaces.

Decomposition strategy
-----------------------
Given trial-averaged activity matrices C1 (n_bins x n_neurons) and
C2 (n_bins x n_neurons) for Context 1 and Context 2 respectively:

    mean    = (C1 + C2) / 2           -- captures pure spatial variance
    diff    = (C1 - C2) / 2           -- captures pure contextual variance

The spatial subspace basis Q_s is found by PCA on `mean` (rows are spatial
bins, columns are neurons).  The contextual subspace basis Q_c is found by
PCA on `diff`.  The principal angles theta_k between the two k-dimensional
subspaces are then:

    cos(theta_k) = singular values of  Q_s.T @ Q_c

A principal angle of 90 deg means the subspaces are perfectly orthogonal;
0 deg means they are perfectly aligned.  The script reports:
  - per-session distributions of all principal angles (scatter + box per region)
  - the mean principal angle across sessions per region (summary bar)
  - a null distribution obtained by random rotation of the contextual subspace

By Guido Meijer
"""
# %% Imports

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from msvr_functions import paths, load_subjects, figure_style

colors, dpi = figure_style()

# %% Settings

N_COMPONENTS = 10        # dimensionality of each subspace (spatial and contextual)
MIN_NEURONS = 10         # minimum number of neurons per region per session
N_NULL = 10             # number of random rotations for null distribution
REGION_ORDER = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']
FIRST_OBJ = 45           # cm
FAR_OBJ = 135            # cm
OBJ_SIZE = 5             # cm

# %% Load data

path_dict = paths(sync=False)
subjects = load_subjects()
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)

# Load residual spike counts over position
data_file = path_dict['google_drive_data_path'] / 'residuals_position_20mms.pickle'
if not data_file.is_file():
    data_file = path_dict['google_drive_data_path'] / 'residuals_position.pickle'

with open(data_file, 'rb') as handle:
    spike_dict = pickle.load(handle)

# %% Helper functions


def principal_angles_deg(Q1, Q2):
    """
    Compute the principal angles (in degrees) between two subspaces defined by
    orthonormal column matrices Q1 (n x p) and Q2 (n x q).

    Returns an array of min(p, q) angles in degrees, sorted ascending.
    The singular values of Q1.T @ Q2 equal the cosines of the principal angles.
    """
    M = Q1.T @ Q2                        # (p, q)
    sv = svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)          # numerical safety before arccos
    angles = np.degrees(np.arccos(sv))
    return np.sort(angles)


def random_rotation_matrix(n, rng):
    """
    Draw a uniformly random orthogonal matrix from the Haar measure on O(n)
    via QR decomposition of a random Gaussian matrix.
    """
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # Correct signs so det(Q) = +1
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def subspace_basis(X, n_components):
    """
    Given a data matrix X (n_obs x n_neurons), z-score each neuron (column),
    fit PCA, and return the orthonormal basis Q (n_neurons x n_components)
    formed by the top PCA components, together with explained variance ratios.

    The PCA components live in neuron space: they are the directions along which
    the observations vary the most. Transposing pca.components_ gives us an
    orthonormal column matrix whose columns span the top PCA subspace in
    n_neurons-dimensional space.
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X_sc)
    # components_ shape: (n_comp, n_neurons) -> transpose to (n_neurons, n_comp)
    Q = pca.components_.T
    return Q, pca.explained_variance_ratio_


# %% Build common position grid and accumulate per-region data

all_positions = []
for pos in spike_dict['position']:
    all_positions.extend(pos)
unique_positions_mm = np.sort(np.unique(all_positions))
unique_positions_cm = unique_positions_mm / 10.0
n_bins = len(unique_positions_mm)

region_activity_c1 = {r: [] for r in REGION_ORDER}
region_activity_c2 = {r: [] for r in REGION_ORDER}

angle_records = []   # one row per (session x region x angle_index)
rng = np.random.default_rng(seed=42)

# %% Main loop: session-level subspace angle computation

for i in range(len(spike_dict['residuals'])):
    this_subject = str(spike_dict['subject'][i])
    this_ses     = str(spike_dict['date'][i])
    this_probe   = str(spike_dict['probe'][i])

    # Only Far sessions (same filter as run_pca_context_per_region.py)
    sub_match = subjects.loc[subjects['SubjectID'].astype(str) == this_subject, 'Far']
    is_far = sub_match.values[0] if len(sub_match) > 0 else np.nan
    if is_far == 0:
        continue

    spatial_bins    = spike_dict['position'][i]
    context_per_bin = spike_dict['context'][i]
    residuals       = spike_dict['residuals'][i]
    regions         = spike_dict['region'][i]

    if regions is None or residuals is None:
        continue

    for region in REGION_ORDER:
        region_mask = (regions == region)
        if np.sum(region_mask) < MIN_NEURONS:
            continue

        spike_counts = residuals[:, region_mask]

        # Remove zero-variance neurons
        valid = np.std(spike_counts, axis=0) > 0
        spike_counts = spike_counts[:, valid]
        if spike_counts.shape[1] < MIN_NEURONS:
            continue

        n_neurons = spike_counts.shape[1]

        # Trial-averaged activity per position bin for each context
        mean_c1 = np.full((n_bins, n_neurons), np.nan)
        mean_c2 = np.full((n_bins, n_neurons), np.nan)

        for b, pos in enumerate(unique_positions_mm):
            mask_1 = (spatial_bins == pos) & (context_per_bin == 1)
            mask_2 = (spatial_bins == pos) & (context_per_bin == 2)
            if np.sum(mask_1) > 0:
                mean_c1[b, :] = np.mean(spike_counts[mask_1, :], axis=0)
            if np.sum(mask_2) > 0:
                mean_c2[b, :] = np.mean(spike_counts[mask_2, :], axis=0)

        if np.isnan(mean_c1).any() or np.isnan(mean_c2).any():
            continue

        # Accumulate for pseudo-population (done outside the loop)
        region_activity_c1[region].append(mean_c1)
        region_activity_c2[region].append(mean_c2)

        # ------------------------------------------------------------------
        # 1) Variance decomposition into pure spatial and pure contextual
        # ------------------------------------------------------------------
        # Spatial component: average across contexts  -> position-driven variance
        X_spatial = (mean_c1 + mean_c2) / 2.0      # (n_bins, n_neurons)

        # Contextual component: difference across contexts -> context-driven variance
        X_context = (mean_c1 - mean_c2) / 2.0      # (n_bins, n_neurons)

        # ------------------------------------------------------------------
        # 2) Independent PCA to extract subspace bases in neuron space
        # ------------------------------------------------------------------
        n_comp = min(N_COMPONENTS, n_neurons - 1, n_bins - 1)
        if n_comp < 2:
            continue

        # Q_s: orthonormal columns spanning the top spatial PCA subspace
        # Q_c: orthonormal columns spanning the top contextual PCA subspace
        Q_s, var_exp_s = subspace_basis(X_spatial, n_comp)   # (n_neurons, n_comp)
        Q_c, var_exp_c = subspace_basis(X_context, n_comp)   # (n_neurons, n_comp)

        # ------------------------------------------------------------------
        # 3) Principal angles between the two subspaces
        # ------------------------------------------------------------------
        angles_deg = principal_angles_deg(Q_s, Q_c)   # sorted, length n_comp
        obs_mean   = float(np.mean(angles_deg))

        # ------------------------------------------------------------------
        # 4) Null distribution: random rotation of contextual basis
        #    Apply a Haar-random orthogonal matrix to rotate Q_c and recompute
        #    angles. Under H0 (orthogonal subspaces) the mean angle should be
        #    close to what we see here; if the observed mean is significantly
        #    smaller, the subspaces are more aligned than expected by chance.
        # ------------------------------------------------------------------
        null_means = np.empty(N_NULL)
        for n_iter in range(N_NULL):
            Rot = random_rotation_matrix(n_neurons, rng)
            Q_c_rand = Rot[:, :n_comp]
            null_means[n_iter] = np.mean(principal_angles_deg(Q_s, Q_c_rand))

        # One-tailed p-value: fraction of null mean angles <= observed mean angle
        p_val = float(np.mean(null_means <= obs_mean))

        for k, ang in enumerate(angles_deg):
            angle_records.append({
                'subject':          this_subject,
                'session':          this_ses,
                'probe':            this_probe,
                'region':           region,
                'is_far':           is_far,
                'angle_index':      k,
                'angle_deg':        ang,
                'mean_angle_deg':   obs_mean,
                'p_value_vs_null':  p_val,
                'var_exp_spatial':  float(np.sum(var_exp_s) * 100),
                'var_exp_context':  float(np.sum(var_exp_c) * 100),
                'n_neurons':        n_neurons,
                'n_components':     n_comp,
            })

angle_df = pd.DataFrame(angle_records)

# %% Pseudo-population level: concatenate all sessions per region

pseudo_results = {}

for region in REGION_ORDER:
    if len(region_activity_c1[region]) == 0:
        continue

    X1 = np.hstack(region_activity_c1[region])   # (n_bins, total_neurons)
    X2 = np.hstack(region_activity_c2[region])   # (n_bins, total_neurons)
    n_tot = X1.shape[1]

    X_spatial = (X1 + X2) / 2.0
    X_context = (X1 - X2) / 2.0

    n_comp = min(N_COMPONENTS, n_tot - 1, n_bins - 1)
    if n_comp < 2:
        continue

    Q_s, var_exp_s = subspace_basis(X_spatial, n_comp)
    Q_c, var_exp_c = subspace_basis(X_context, n_comp)
    angles_deg     = principal_angles_deg(Q_s, Q_c)
    obs_mean       = float(np.mean(angles_deg))

    null_means_pp = np.empty(N_NULL)
    for n_iter in range(N_NULL):
        Rot = random_rotation_matrix(n_tot, rng)
        null_means_pp[n_iter] = np.mean(principal_angles_deg(Q_s, Rot[:, :n_comp]))

    pseudo_results[region] = {
        'angles_deg':      angles_deg,
        'mean_angle_deg':  obs_mean,
        'null_mean':       float(np.mean(null_means_pp)),
        'null_std':        float(np.std(null_means_pp)),
        'null_angles':     null_means_pp,
        'p_value':         float(np.mean(null_means_pp <= obs_mean)),
        'var_exp_spatial': float(np.sum(var_exp_s) * 100),
        'var_exp_context': float(np.sum(var_exp_c) * 100),
        'n_neurons':       n_tot,
        'n_components':    n_comp,
    }

# =========================================================================
# FIGURES
# =========================================================================

save_dir = path_dict['paper_fig_path'] / 'Population'
save_dir.mkdir(parents=True, exist_ok=True)

# %% Figure 1: Principal angle spectrum per brain region (pseudo-population)
# Each bar = one principal angle; dashed line = null mean +/- 1 SD shading.

regions_with_data = [r for r in REGION_ORDER if r in pseudo_results]
n_r = len(regions_with_data)

f, axs = plt.subplots(1, n_r, figsize=(1.8 * n_r, 2.2), dpi=dpi)
if n_r == 1:
    axs = [axs]

for ax_idx, region in enumerate(regions_with_data):
    res   = pseudo_results[region]
    ang   = res['angles_deg']
    nc    = res['n_components']
    nm    = res['null_mean']
    ns    = res['null_std']
    pv    = res['p_value']
    rc    = colors.get(region, 'k')
    ax    = axs[ax_idx]
    x_pos = np.arange(1, nc + 1)

    ax.bar(x_pos, ang, color=rc, alpha=0.8, width=0.6)

    ax.axhline(nm, color='grey', ls='--', lw=1.0, label='Null mean')
    ax.fill_between([0.5, nc + 0.5], nm - ns, nm + ns,
                    color='grey', alpha=0.2, label='Null +/-1 SD')
    ax.axhline(90, color='k', ls=':', lw=0.7, label='90 deg (orthogonal)')

    ax.set_title(f'{region}\n(N={res["n_neurons"]})', color=rc,
                 weight='bold', fontsize=7)
    ax.set_xlabel('Principal angle #', labelpad=2)
    if ax_idx == 0:
        ax.set_ylabel('Angle (deg)', labelpad=2)
        ax.legend(frameon=False, fontsize=5, loc='lower right')
    ax.set_xticks(x_pos)
    ax.set_ylim(0, 100)
    ax.set_xlim(0.3, nc + 0.7)
    ax.text(0.97, 0.05, f'p={pv:.3f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=6)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(save_dir / 'subspace_angles_pseudo_population.pdf')
plt.savefig(save_dir / 'subspace_angles_pseudo_population.jpg', dpi=600)
plt.show()

# %% Figure 2: Per-session mean principal angle per region (box + strip)

ses_summary = (angle_df
               .groupby(['subject', 'session', 'probe', 'region'])['angle_deg']
               .mean()
               .reset_index()
               .rename(columns={'angle_deg': 'mean_angle_deg'}))

f, ax = plt.subplots(figsize=(0.9 * len(REGION_ORDER), 2.2), dpi=dpi)

sns.boxplot(data=ses_summary, x='region', y='mean_angle_deg',
            order=REGION_ORDER, palette=colors, hue='region',
            linewidth=0.75, fliersize=0, width=0.5, ax=ax)
sns.stripplot(data=ses_summary, x='region', y='mean_angle_deg',
              order=REGION_ORDER, palette=colors, hue='region',
              size=3, alpha=0.6, jitter=True, ax=ax, legend=False)
ax.axhline(90, color='k', ls=':', lw=0.7, label='90 deg (orthogonal)')
ax.set_ylabel('Mean principal angle (deg)', labelpad=2)
ax.set_xlabel('')
ax.set_ylim(0, 100)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(save_dir / 'subspace_angles_per_session.pdf')
plt.savefig(save_dir / 'subspace_angles_per_session.jpg', dpi=600)
plt.show()

# %% Figure 3: Full angle spectrum per session (thin lines) + mean, per region

f, axs = plt.subplots(1, len(REGION_ORDER), figsize=(1.6 * len(REGION_ORDER), 2.0),
                      dpi=dpi, sharey=True)
if len(REGION_ORDER) == 1:
    axs = [axs]

for i, region in enumerate(REGION_ORDER):
    sub = angle_df[angle_df['region'] == region]
    if sub.empty:
        axs[i].axis('off')
        continue

    rc = colors.get(region, 'k')

    for (ses, probe), grp in sub.groupby(['session', 'probe']):
        g = grp.sort_values('angle_index')
        axs[i].plot(g['angle_index'] + 1, g['angle_deg'],
                    color=rc, alpha=0.3, lw=0.8)

    mean_idx = sub.groupby('angle_index')['angle_deg'].mean()
    axs[i].plot(mean_idx.index + 1, mean_idx.values,
                color=rc, lw=2.0, label='Session mean')

    if region in pseudo_results:
        nm = pseudo_results[region]['null_mean']
        ns = pseudo_results[region]['null_std']
        nc = pseudo_results[region]['n_components']
        axs[i].axhline(nm, color='grey', ls='--', lw=0.8, label='Null mean')
        axs[i].fill_between([0.5, nc + 0.5], nm - ns, nm + ns,
                             color='grey', alpha=0.15)

    axs[i].axhline(90, color='k', ls=':', lw=0.7)
    axs[i].set_title(region, color=rc, weight='bold', fontsize=7)
    axs[i].set_xlabel('Principal angle #', labelpad=2)
    if i == 0:
        axs[i].set_ylabel('Angle (deg)', labelpad=2)
    axs[i].set_ylim(0, 100)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(save_dir / 'subspace_angles_spectrum_per_region.pdf')
plt.savefig(save_dir / 'subspace_angles_spectrum_per_region.jpg', dpi=600)
plt.show()

# %% Figure 4: Summary bar -- observed vs null mean angle per region

obs_means  = [pseudo_results[r]['mean_angle_deg'] for r in regions_with_data]
null_means = [pseudo_results[r]['null_mean']      for r in regions_with_data]
null_stds  = [pseudo_results[r]['null_std']       for r in regions_with_data]
p_vals     = [pseudo_results[r]['p_value']        for r in regions_with_data]

x     = np.arange(len(regions_with_data))
width = 0.35

f, ax = plt.subplots(figsize=(1.2 * len(regions_with_data), 2.2), dpi=dpi)

ax.bar(x - width / 2, null_means, width,
       yerr=null_stds, color='lightgrey', edgecolor='k', lw=0.7,
       capsize=3, label='Null (random rotation)', error_kw={'lw': 0.8})

bar_colors = [colors.get(r, 'steelblue') for r in regions_with_data]
ax.bar(x + width / 2, obs_means, width,
       color=bar_colors, edgecolor='k', lw=0.7, label='Observed')

for j, (pv, xj, obs) in enumerate(zip(p_vals, x, obs_means)):
    stars = ('***' if pv < 0.001 else '**' if pv < 0.01
             else '*' if pv < 0.05 else 'n.s.')
    ax.text(xj + width / 2, obs + 1.5, stars, ha='center', va='bottom', fontsize=7)

ax.axhline(90, color='k', ls=':', lw=0.7, label='90 deg (orthogonal)')
ax.set_xticks(x)
ax.set_xticklabels(regions_with_data, rotation=45, ha='right')
ax.set_ylabel('Mean principal angle (deg)', labelpad=2)
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=5.5, loc='lower right')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(save_dir / 'subspace_angles_summary_bar.pdf')
plt.savefig(save_dir / 'subspace_angles_summary_bar.jpg', dpi=600)
plt.show()

# %% Print summary table

print('\n=== Subspace orthogonality summary (pseudo-population) ===')
print(f'{"Region":<8} {"N neurons":>10} {"n_comp":>7} '
      f'{"Obs angle":>11} {"Null mean":>11} {"Null SD":>9} {"p-value":>9}')
print('-' * 72)
for region in REGION_ORDER:
    if region not in pseudo_results:
        continue
    res = pseudo_results[region]
    print(f'{region:<8} {res["n_neurons"]:>10} {res["n_components"]:>7} '
          f'{res["mean_angle_deg"]:>10.2f}  {res["null_mean"]:>10.2f} '
          f'{res["null_std"]:>9.2f} {res["p_value"]:>9.4f}')
