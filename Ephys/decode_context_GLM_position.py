# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from scipy.stats import binned_statistic
from joblib import Parallel, delayed
from brainbox.population.decode import classify
from msvr_functions import paths, load_neural_data, load_objects, bin_continuous_signal

# Settings
BIN_SIZE = 50  # mm
STEP_SIZE = 5
MIN_TRIALS = 30
MIN_SPEED = 0 # mm/s
N_CORES = -2

# Create position bins
rel_bin_centers = np.arange((BIN_SIZE/2), 1500 - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
kfold_cv = KFold(n_splits=10, shuffle=True, random_state=42)
clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=1))
scaler = StandardScaler()

# Function for parallel processing
def decode_context(bin_center, X_decode, y, X_bin_centers):

    this_X = X_decode[X_bin_centers == bin_center, :]
    this_y = y[X_bin_centers == bin_center]
    accuracy, _, _ = classify(this_X, this_y, clf, cross_validation=kfold_cv)

    return accuracy


# Loop over recordings
glm_results = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\n{subject} {date} {probe} ({i} of {rec.shape[0]})')

    # Load in data
    session_path = path_dict['local_data_path'] / 'subjects' / f'{subject}' / f'{date}'
    trials = pd.read_csv(session_path / 'trials.csv')
    if trials.shape[0] < MIN_TRIALS:
        print('Too few trials, skipping session')
        continue
    spikes, clusters, channels = load_neural_data(session_path, probe)
    position = np.load(session_path / 'continuous.wheelDistance.npy')
    running_speed = np.load(session_path / 'continuous.wheelSpeed.npy')
    timestamps = np.load(session_path / 'continuous.times.npy')
    lick_pos = np.load(session_path / 'lick.positions.npy')
    sampling_freq = 1 / np.round(np.mean(np.diff(timestamps)), 3)
    all_obj_df = load_objects(subject, date)

    # Get the acceleration as the first derivative of the speed
    acceleration = np.gradient(running_speed)

    # Filter spikes by speed
    valid_spike_mask = spikes['speeds'] > MIN_SPEED

    # Create a dictionary of valid spike POSITIONS, keyed by neuron_id
    print("Pre-filtering spike times by neuron and speed...")
    spikes_by_neuron = {}
    for neuron_id in clusters['cluster_id']:
        # Get spikes for this neuron that also pass the speed threshold
        neuron_mask = np.isin(spikes['clusters'], neuron_id) & valid_spike_mask
        spikes_by_neuron[neuron_id] = spikes['distances'][neuron_mask]

    # Pre-initialize the output dictionary
    binned_spikes = {f'neuron{neuron_id}': [] for neuron_id in clusters['cluster_id']}

    ## 2. MAIN LOOP

    print('Building predictor matrix...')
    time_per_bin_list, speed_pred, acc_pred, lick_pred, trial_id_list = [], [], [], [], []
    rel_pos_list = []

    for row in trials.itertuples():

        # Define spatial bins
        abs_bin_centers = rel_bin_centers + row.enterEnvPos
        rel_pos_list.append(rel_bin_centers)

        # Get trial id
        trial_id_list.append(np.full(abs_bin_centers.shape[0], row.Index))

        # Speed predictor
        this_speed = bin_continuous_signal(position, running_speed, abs_bin_centers, BIN_SIZE)
        speed_pred.append(this_speed)
        this_acc = bin_continuous_signal(position, acceleration, abs_bin_centers, BIN_SIZE)
        acc_pred.append(this_acc)

        # Licks
        this_lick = bin_continuous_signal(position, lick_pos, abs_bin_centers, BIN_SIZE, statistic='count')
        lick_pred.append(this_lick)

        # Time occupancy
        this_counts = bin_continuous_signal(position, timestamps, abs_bin_centers, BIN_SIZE, statistic='count')
        time_per_bin_list.append(this_counts / sampling_freq)

        # --- Get binned spiking activity per neuron ---
        for neuron_id, neuron_spike_pos in spikes_by_neuron.items():

            # Filter spikes for this trial (by position range)
            # They are already filtered by speed globally

            # We need spikes that are within [start_pos, end_pos]
            # Note: neuron_spike_pos are absolute positions
            trial_spike_mask = ((neuron_spike_pos >= row.enterEnvPos)
                                & (neuron_spike_pos <= row.exitEnvPos))
            these_spike_pos = neuron_spike_pos[trial_spike_mask]

            # Bin spikes
            these_spikes, _ = np.histogram(these_spike_pos, bins=bin_edges)

            # Append
            binned_spikes[f'neuron{neuron_id}'].append(these_spikes)

    print('Matrix build complete.')

    # --- Final Step (Post-processing) ---
    exposure_vector = np.concatenate(time_per_bin_list)
    trial_id = np.concatenate(trial_id_list)
    final_predictors = {
        'speed': scaler.fit_transform(np.concatenate(speed_pred).reshape(-1, 1)).T[0],
        'acceleration': scaler.fit_transform(np.concatenate(acc_pred).reshape(-1, 1)).T[0],
        'lick': np.concatenate(lick_pred),
        'trial_id': trial_id
    }

    # Add neural data
    for neuron_key, spike_list in binned_spikes.items():
        final_predictors[neuron_key] = np.concatenate(spike_list)

    # Convert to pandas dataframe
    df = pd.DataFrame(final_predictors)

    # Get the base X matrix and add the constant
    X_base = df[['speed', 'acceleration', 'lick']]
    X_base_with_const = sm.add_constant(X_base, prepend=False)

    # Fit GLM
    print("Fitting GLM...")
    X_decode = []
    neuron_columns = [col for col in df.columns if col.startswith('neuron')]
    for neuron_col in neuron_columns:

        # Fit GLM
        glm_nb = sm.GLM(
            df[neuron_col],
            X_base_with_const,
            family=sm.families.Poisson(),
            exposure=exposure_vector
            )

        # Get residuals
        try:
            results = glm_nb.fit()
            X_decode.append(results.resid_response.values / exposure_vector)

        except Exception as e:
            print(f"Failed to fit model for {neuron_col}: {e}")
            X_decode.append(np.zeros(df[neuron_col].shape))

    print("GLM fitting complete.")

    # Finalize
    X_decode = np.vstack(X_decode).T  # bins x neurons
    all_rel_position = np.concatenate(rel_pos_list)
    neuron_ids = np.array([int(i[6:]) for i in neuron_columns])
    assert all(neuron_ids == clusters['cluster_id'])

    # --- Done with the GLM, on to the decoding ---

    # Get the context id per trial
    all_y = np.array([trials.loc[i, 'soundId'] for i in trial_id])

    # Get position bins
    rel_pos_bins = np.unique(all_rel_position)
    rel_pos_bins = rel_pos_bins[rel_pos_bins < 1500]  # exclude out of bounds bins

    # Decode per brain region
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        print(f'Decoding {region}')

        # Select neurons from this brain region
        X_region = X_decode[:, clusters['region'] == region]

        # Do decoding per position bin
        results = Parallel(n_jobs=N_CORES)(
            delayed(decode_context)(bin_center, X_region, all_y, all_rel_position)
            for bin_center in rel_pos_bins)
        accuracy = [i for i in results]

        asd

# Save to disk
glm_results.to_csv(path_dict['save_path'] / 'GLM_results_position.csv', index=False)
