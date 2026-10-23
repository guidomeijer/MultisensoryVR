# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 2026

Same as create_position_residuals.py but neural activity is binned in time bins instead of
spatial bins. Only the periods during which the mouse is inside an environment (between
enterEnvTime and exitEnvTime of each trial) are used.

Position is regressed out by evaluating a B-spline basis on the (mean) position of the mouse,
relative to enterEnvPos, within each time bin. Because the spline knots are fixed over the
full track length, every time bin gets spline predictors that depend only on where the
mouse was during that bin, which makes the position term equivalent to the one used for the
spatial bins.

By Guido Meijer
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pickle
from patsy import dmatrix
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, bin_signal

# Settings
BIN_SIZE = 0.1  # s
TRACK_LENGTH = 1500  # mm
MIN_TRIALS = 0
MIN_SPEED = 20  # mm/s, time bins with a lower mean speed are excluded
N_CORES = -8
OVERWRITE = True


def process_recording(i, subject, date, probe, n_recs, path_dict):
    print(f'\n{subject} {date} {probe} ({i} of {n_recs})')

    # Initialize
    scaler = StandardScaler()

    # Load in data
    session_path = path_dict['local_data_path'] / 'subjects' / f'{subject}' / f'{date}'
    trials = pd.read_csv(session_path / 'trials.csv')
    if trials.shape[0] < MIN_TRIALS:
        print('Too few trials, skipping session')
        return None

    spikes, clusters, channels = load_neural_data(session_path, probe)
    position = np.load(session_path / 'continuous.wheelDistance.npy')
    running_speed = np.load(session_path / 'continuous.wheelSpeed.npy')
    timestamps = np.load(session_path / 'continuous.times.npy')
    lick_times = np.load(session_path / 'lick.times.npy')

    # Get the acceleration as the first derivative of the speed
    acceleration = np.gradient(running_speed, timestamps)

    # Create time bins for every trial, only for the periods the mouse is in the environment
    # (between enterEnvTime and exitEnvTime). The inter-trial periods between exitEnvTime and
    # the next enterEnvTime are thereby excluded, and the last bin ends before the exitEnvTime
    # of the last trial.
    bin_centers_list, trial_id_list = [], []
    for row in trials.itertuples():
        if row.soundId == 0:
            continue
        n_bins = int(np.floor((row.exitEnvTime - row.enterEnvTime) / BIN_SIZE))
        bin_centers_list.append(row.enterEnvTime + (BIN_SIZE / 2) + np.arange(n_bins) * BIN_SIZE)
        trial_id_list.append(np.full(n_bins, row.Index))
    bin_centers = np.concatenate(bin_centers_list)
    trial_id = np.concatenate(trial_id_list)

    # Get behavioral variables per time bin
    print('Building predictor matrix...')
    speed = bin_signal(timestamps, running_speed, bin_centers, BIN_SIZE)
    acc = bin_signal(timestamps, acceleration, bin_centers, BIN_SIZE)
    abs_pos = bin_signal(timestamps, position, bin_centers, BIN_SIZE)
    licks = bin_signal(None, lick_times, bin_centers, BIN_SIZE, statistic='count')
    rel_pos = abs_pos - trials.loc[trial_id, 'enterEnvPos'].values
    time_in_trial = bin_centers - trials.loc[trial_id, 'enterEnvTime'].values

    # Exclude time bins without behavioral samples or in which the mouse was too slow
    keep = ~np.isnan(speed) & ~np.isnan(acc) & ~np.isnan(abs_pos) & (speed > MIN_SPEED)
    bin_centers, trial_id = bin_centers[keep], trial_id[keep]
    speed, acc, licks = speed[keep], acc[keep], licks[keep]
    rel_pos, time_in_trial = rel_pos[keep], time_in_trial[keep]
    print(f'{np.sum(keep)} of {keep.shape[0]} time bins included')

    # Construct dataframe with predictors
    df = pd.DataFrame({
        'speed': scaler.fit_transform(speed.reshape(-1, 1)).T[0],
        'acceleration': scaler.fit_transform(acc.reshape(-1, 1)).T[0],
        'lick': licks,
        'trial_id': trial_id
    })

    # Get binned spike counts per neuron
    print('Binning spikes...')
    neuron_columns = [f'neuron{neuron_id}' for neuron_id in clusters['cluster_id']]
    spike_counts = {}
    for neuron_id, neuron_col in zip(clusters['cluster_id'], neuron_columns):
        spike_counts[neuron_col] = bin_signal(
            None, spikes['times'][spikes['clusters'] == neuron_id], bin_centers, BIN_SIZE,
            statistic='count')
    df = pd.concat([df, pd.DataFrame(spike_counts)], axis=1)

    # Position splines evaluated at the position of the mouse within each time bin.
    # Knots are fixed over the full track so the basis is the same for every recording.
    spatial_basis = dmatrix(
        f"bs(pos, df=10, include_intercept=False, lower_bound=0, upper_bound={TRACK_LENGTH}) - 1",
        {"pos": np.clip(rel_pos, 0, TRACK_LENGTH)},
        return_type='dataframe'
    )
    spatial_basis.columns = [f'spatial_basis_{i}' for i in range(spatial_basis.shape[1])]

    # Get the base X matrix and add the constant
    X_motor = df[['speed', 'acceleration', 'lick']]
    X_motor_with_pos = pd.concat([X_motor, spatial_basis.reset_index(drop=True)], axis=1)
    X_full = sm.add_constant(X_motor_with_pos, prepend=False)

    # Fit GLM (all bins have the same duration so no exposure is needed)
    residual_arr, glm_results = [], []
    for neuron_col in neuron_columns:

        glm_nb = sm.GLM(
            df[neuron_col],
            X_full,
            family=sm.families.Poisson()
            )

        # Get residuals and convert from spike counts to firing rate (spikes/s)
        try:
            results = glm_nb.fit()
            residual_arr.append(results.resid_response.values / BIN_SIZE)
            these_results = results.params
            these_results['pseudo_rsquared'] = results.pseudo_rsquared()
            glm_results.append(these_results)
        except Exception as e:
            print(f"Failed to fit model for {neuron_col}: {e}")
            residual_arr.append(np.zeros(df[neuron_col].shape))

    # Finalize
    residual_arr = np.vstack(residual_arr).T  # bins x neurons
    neuron_ids = np.array([int(i[6:]) for i in neuron_columns])
    assert all(neuron_ids == clusters['cluster_id'])

    return {
        'residuals': residual_arr,
        'neuron_id': neuron_ids,
        'region': clusters['region'],
        'acronym': clusters['acronym'],
        'time': bin_centers,
        'time_in_trial': time_in_trial,
        'position': rel_pos,
        'speed': speed,
        'trial': trial_id,
        'context': trials.loc[trial_id, 'soundId'].values,
        'glm_results': glm_results,
        'subject': subject,
        'date': date,
        'probe': probe
    }


if __name__ == '__main__':
    # Initialize
    path_dict = paths(sync=False)
    rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
    save_path = path_dict['google_drive_data_path'] / f'residuals_time_{MIN_SPEED}mms.pickle'

    if OVERWRITE:
        residuals_dict = {
            'residuals': [],
            'neuron_id': [],
            'region': [],
            'acronym': [],
            'time': [],
            'time_in_trial': [],
            'position': [],
            'speed': [],
            'trial': [],
            'context': [],
            'glm_results': [],
            'subject': [],
            'date': [],
            'probe': []
            }
    else:
        with open(save_path, 'rb') as handle:
            residuals_dict = pickle.load(handle)
        match_cols = ['subject', 'date', 'probe']
        res_df = pd.DataFrame({c: residuals_dict[c] for c in match_cols}).astype(str)
        rec[match_cols] = rec[match_cols].astype(str)
        merged = rec.merge(res_df, on=match_cols, how='left', indicator=True)
        rec = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

    # Loop over recordings
    print(f'Processing {rec.shape[0]} recordings')
    results = Parallel(n_jobs=N_CORES)(
        delayed(process_recording)(i, subject, date, probe, rec.shape[0], path_dict)
        for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe']))
    )

    # Add results to dictionary
    for result in results:
        if result is not None:
            for key in residuals_dict.keys():
                residuals_dict[key].append(result[key])

    # Save to disk
    with open(save_path, 'wb') as handle:
        pickle.dump(residuals_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
