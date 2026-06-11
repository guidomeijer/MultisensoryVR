# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:36:00 2026 by Guido Meijer

Use PLS regression to find the low-dimensional projection of cortical activity (VIS, PERI, TEa,
AUD, LEC) that best predicts CA1 activity. Map CA1 activity onto the cortical-driven axis via
dot product along the neuron dimension. Extract per-object CA1 trajectories for rewarded and
non-rewarded trials (skipping object 3 which is never rewarded). Also compute the subspace
orthogonality between all pairs of cortical axis vectors mapping onto CA1.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
TIME_WIN = {'obj1': [-2, 2], 'obj2': [-2, 2]}
BIN_SIZE = 0.025
SMOOTHING = 0.05
MIN_NEURONS = 5  # per region
N_COMPONENTS = 2
N_SPLITS = 10
SUBTRACT_MEAN = False
CORTICAL_REGIONS = ['VIS', 'PERI', 'TEa', 'AUD', 'LEC']
TARGET_REGION = 'CA1'

# Initialize
path_dict = paths()
subjects = load_subjects()

# Load in data
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv', dtype={'subject': str, 'date': str})
rec = rec.drop_duplicates(subset=['subject', 'date'])


# %% Functions

def fit_pls_cortex_to_ca1(x_cortex, y_ca1, n_components, n_splits=10):
    """
    Fit PLS regression from cortical activity (X) to CA1 activity (Y) with cross-validation.

    Returns the weight vector(s) mapping CA1 neurons onto the cortical-driven axis
    (Y loadings from PLS), and the CA1 trajectories projected onto this axis.

    Parameters
    ----------
    x_cortex : 3D array (trials, timebins, cortical_neurons)
    y_ca1 : 3D array (trials, timebins, ca1_neurons)
    n_components : int
    n_splits : int

    Returns
    -------
    ca1_projections : 3D array (trials, timebins, n_components)
        CA1 activity projected onto the cortical-driven axes.
    y_weights_avg : 2D array (ca1_neurons, n_components)
        Average Y weights across CV folds — the axis vectors mapping cortical drive onto CA1.
    """
    n_trials, n_timebins, n_neurons_X = x_cortex.shape
    _, _, n_neurons_Y = y_ca1.shape

    ca1_projections = np.zeros((n_trials, n_timebins, n_components))
    y_weights_all = []

    # K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_idx, test_idx in kf.split(np.arange(n_trials)):

        # Slice
        X_train_3D = x_cortex[train_idx]
        Y_train_3D = y_ca1[train_idx]
        X_test_3D = x_cortex[test_idx]
        Y_test_3D = y_ca1[test_idx]

        # Flatten to 2D: (trials * time, neurons)
        X_train_2D = X_train_3D.reshape(-1, n_neurons_X)
        Y_train_2D = Y_train_3D.reshape(-1, n_neurons_Y)
        Y_test_2D = Y_test_3D.reshape(-1, n_neurons_Y)

        # Apply PCA to keep components explaining 90% of the variance
        pca_X = PCA(n_components=0.90)
        X_train_pca = pca_X.fit_transform(X_train_2D)

        # Determine actual number of PLS components
        actual_components = min(n_components, X_train_pca.shape[1], Y_train_2D.shape[1])

        # Fit PLS regression: cortex predicts CA1
        pls = PLSRegression(n_components=actual_components)
        pls.fit(X_train_pca, Y_train_2D)

        # Store Y weights (ca1_neurons x components)
        y_weights_all.append(pls.y_weights_)

        # Project held-out CA1 activity onto the cortical-driven axis
        # dot product along the neuron dimension: (timebins, neurons) @ (neurons, components)
        n_test_trials = len(test_idx)
        test_projection = Y_test_2D @ pls.y_weights_
        ca1_projections[test_idx, :, :actual_components] = test_projection.reshape(
            n_test_trials, n_timebins, actual_components)

    # Average Y weights across folds
    y_weights_avg = np.mean(np.stack(y_weights_all), axis=0)

    return ca1_projections, y_weights_avg


def compute_subspace_angle(w1, w2):
    """
    Compute the principal angle (in degrees) between two subspaces defined by weight matrices.
    Each weight matrix has shape (neurons, n_components).
    Uses the cosine of the largest singular value of Q1^T @ Q2 where Q1, Q2 are orthonormal bases.

    Returns the angle in degrees (0 = identical subspaces, 90 = orthogonal).
    """
    # Get orthonormal bases
    Q1, _ = np.linalg.qr(w1)
    Q2, _ = np.linalg.qr(w2)

    # Singular values of the cross-product give cosines of principal angles
    svd_vals = np.linalg.svd(Q1.T @ Q2, compute_uv=False)

    # Clamp to avoid numerical issues
    cos_angle = np.clip(svd_vals[0], -1, 1)
    angle_deg = np.degrees(np.arccos(cos_angle))

    return angle_deg


# %% Main loop
trajectory_df = pd.DataFrame()
orthogonality_df = pd.DataFrame()

for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'Processing session {i+1} of {rec.shape[0]} ({subject} {date})')

    # Load in neural data for all probes
    session_path = path_dict['local_data_path'] / 'Subjects' / str(subject) / str(date)
    spikes, clusters, channels = load_multiple_probes(session_path)

    # Load in object entry times
    all_obj_df = load_objects(subject, date)

    # Get binned spike counts per region per object (skip obj3)
    spikes_dict = {'obj1': dict(), 'obj2': dict()}
    for k, probe in enumerate(spikes.keys()):
        for j, region in enumerate(np.unique(clusters[probe]['region'])):
            if region == 'root':
                continue

            # Select neurons
            region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]

            # Loop over objects (only obj1 and obj2)
            for m, obj in enumerate(['obj1', 'obj2']):

                # Get spike counts (trials x neurons x time)
                peth, binned_spikes = calculate_peths(
                    spikes[probe]['times'], spikes[probe]['clusters'], region_neurons,
                    all_obj_df.loc[all_obj_df['object'] == m+1, 'times'].values,
                    np.abs(TIME_WIN[obj][0]), TIME_WIN[obj][1], BIN_SIZE, SMOOTHING,
                    return_fr=False)

                # Reshape into (trials x time x neurons)
                spikes_transposed = np.swapaxes(binned_spikes, 1, 2)

                # Drop silent neurons
                spikes_dropped = spikes_transposed[:, :, np.max(peth['means'], axis=1) > 0.01]
                if spikes_dropped.shape[2] < MIN_NEURONS:
                    continue

                # Subtract the mean over trials from each trial to leave the residuals
                if SUBTRACT_MEAN:
                    spikes_dict[obj][region] = spikes_dropped - np.mean(spikes_dropped, axis=0)
                else:
                    spikes_dict[obj][region] = spikes_dropped

    # Loop over objects
    for m, obj in enumerate(['obj1', 'obj2']):

        # Check that CA1 is available for this object
        if TARGET_REGION not in spikes_dict[obj]:
            continue

        # Get CA1 data
        ca1_data = spikes_dict[obj][TARGET_REGION]  # (trials, time, ca1_neurons)
        time_ax = np.arange(ca1_data.shape[1]) * BIN_SIZE + TIME_WIN[obj][0] + (BIN_SIZE / 2)

        # Get reward labels per trial (goal = rewarded for this object)
        trial_goals = all_obj_df.loc[all_obj_df['object'] == m+1, 'goal'].values

        # Collect cortical-driven axes for subspace orthogonality
        cortex_weights = dict()

        # Loop over cortical regions
        for ctx_region in CORTICAL_REGIONS:
            if ctx_region not in spikes_dict[obj]:
                continue

            cortex_data = spikes_dict[obj][ctx_region]  # (trials, time, ctx_neurons)

            # Run PLS regression: cortex -> CA1
            ca1_proj, y_weights = fit_pls_cortex_to_ca1(
                cortex_data, ca1_data, n_components=N_COMPONENTS, n_splits=N_SPLITS)

            # Store weights for orthogonality computation
            cortex_weights[ctx_region] = y_weights

            # Get CA1 trajectories per object for rewarded vs non-rewarded
            for reward_label, reward_name in zip([1, 0], ['rewarded', 'non-rewarded']):
                trial_mask = trial_goals == reward_label
                if np.sum(trial_mask) == 0:
                    continue

                # Average trajectory across matching trials, per component
                for comp in range(N_COMPONENTS):
                    mean_traj = np.mean(ca1_proj[trial_mask, :, comp], axis=0)

                    trajectory_df = pd.concat([trajectory_df, pd.DataFrame(data={
                        'ca1_projection': mean_traj,
                        'time': time_ax,
                        'component': comp + 1,
                        'cortical_region': ctx_region,
                        'reward': reward_name,
                        'object': obj,
                        'subject': subject,
                        'date': date
                    })], ignore_index=True)

        # Compute subspace orthogonality between all pairs of cortical region axes
        ctx_pairs = list(combinations(sorted(cortex_weights.keys()), 2))
        for pair in ctx_pairs:
            angle = compute_subspace_angle(cortex_weights[pair[0]], cortex_weights[pair[1]])
            orthogonality_df = pd.concat([orthogonality_df, pd.DataFrame(
                index=[orthogonality_df.shape[0]], data={
                    'angle_deg': angle,
                    'region_pair': f'{pair[0]}-{pair[1]}',
                    'region_1': pair[0],
                    'region_2': pair[1],
                    'object': obj,
                    'subject': subject,
                    'date': date
                })], ignore_index=True)

    # Save to disk after each session
    trajectory_df.to_csv(
        path_dict['google_drive_data_path'] / 'cortex_to_ca1_trajectories.csv', index=False)
    orthogonality_df.to_csv(
        path_dict['google_drive_data_path'] / 'cortex_to_ca1_orthogonality.csv', index=False)

print('Done!')
