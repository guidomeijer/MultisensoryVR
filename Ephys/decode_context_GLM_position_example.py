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
import matplotlib.pyplot as plt
from patsy import dmatrix
import seaborn as sns
from msvr_functions import paths, load_neural_data, load_objects, bin_signal, figure_style

# Session selection
subject = '462910'
date = '20240813'
probe = 'probe01'

# Settings
PLOT_TRIAL = 12
PLOT_NEURON = 143
BIN_SIZE = 50  # mm
STEP_SIZE = 10
MIN_TRIALS = 30
MIN_SPEED = 0 # mm/s

# Create position bins
rel_bin_centers = np.arange((BIN_SIZE/2), 1500 - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
kfold_cv = KFold(n_splits=10, shuffle=True, random_state=42)
clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=1))
scaler = StandardScaler()

# Load in data
session_path = path_dict['local_data_path'] / 'subjects' / f'{subject}' / f'{date}'
trials = pd.read_csv(session_path / 'trials.csv')
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
spikes_by_neuron = {}
for neuron_id in clusters['cluster_id']:
    # Get spikes for this neuron that also pass the speed threshold
    neuron_mask = np.isin(spikes['clusters'], neuron_id) & valid_spike_mask
    spikes_by_neuron[neuron_id] = spikes['distances'][neuron_mask]

## 2. MAIN LOOP

binned_spikes = {f'neuron{neuron_id}': [] for neuron_id in clusters['cluster_id']}
time_per_bin_list, speed_pred, acc_pred, lick_pred, trial_id_list = [], [], [], [], []
rel_pos_list = []

for row in trials.itertuples():

    # Define spatial bins
    abs_bin_centers = rel_bin_centers + row.enterEnvPos
    rel_pos_list.append(rel_bin_centers)

    # Get trial id
    trial_id_list.append(np.full(abs_bin_centers.shape[0], row.Index))

    # Speed predictor
    this_speed = bin_signal(position, running_speed, abs_bin_centers, BIN_SIZE)
    speed_pred.append(this_speed)
    this_acc = bin_signal(position, acceleration, abs_bin_centers, BIN_SIZE)
    acc_pred.append(this_acc)

    # Licks
    this_lick = bin_signal(position, lick_pos, abs_bin_centers, BIN_SIZE, statistic='count')
    lick_pred.append(this_lick)

    # Time occupancy
    dt = np.concatenate((np.diff(timestamps), [np.mean(np.diff(timestamps))]))
    this_occupancy = bin_signal(position, dt, abs_bin_centers, BIN_SIZE,
                                        statistic='sum')
    time_per_bin_list.append(this_occupancy)

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
        these_binned_spikes = bin_signal(position, these_spike_pos, abs_bin_centers, BIN_SIZE,
                                         statistic='count')
        
        # Append
        binned_spikes[f'neuron{neuron_id}'].append(these_binned_spikes)

# %% --- Final Step (Post-processing) ---

exposure_vector = np.concatenate(time_per_bin_list)
trial_id = np.concatenate(trial_id_list)
final_predictors = {
    'speed': scaler.fit_transform(np.concatenate(speed_pred).reshape(-1, 1)).T[0],
    'acceleration': scaler.fit_transform(np.concatenate(acc_pred).reshape(-1, 1)).T[0],
    'lick': np.concatenate(lick_pred),
    'trial_id': trial_id
}

# Convert to pandas dataframe
df = pd.DataFrame(final_predictors)

# Add position splines
all_rel_positions = np.concatenate(rel_pos_list)
spatial_basis = dmatrix(
    "bs(pos, df=10, include_intercept=False) - 1", 
    {"pos": all_rel_positions}, 
    return_type='dataframe'
)
spatial_basis.columns = [f'spatial_basis_{i}' for i in range(spatial_basis.shape[1])]

# Get the base X matrix and add the constant
X_motor = df[['speed', 'acceleration', 'lick']]
X_motor_with_pos = pd.concat([X_motor, spatial_basis.reset_index(drop=True)], axis=1)
X_full = sm.add_constant(X_motor_with_pos, prepend=False)

# Fit GLM
print("Fitting GLM...")
spike_resid = {f'neuron{neuron_id}': [] for neuron_id in clusters['cluster_id']}
spike_predict = {f'neuron{neuron_id}': [] for neuron_id in clusters['cluster_id']}
speed_coef = {f'neuron{neuron_id}': [] for neuron_id in clusters['cluster_id']}
all_results = {f'neuron{neuron_id}': [] for neuron_id in clusters['cluster_id']}
for neuron_id in clusters['cluster_id']:
    if np.sum(np.concatenate(binned_spikes[f'neuron{neuron_id}'])) < 10:
        continue
    glm_nb = sm.GLM(
        np.concatenate(binned_spikes[f'neuron{neuron_id}']),
        X_full,
        family=sm.families.Poisson(),
        exposure=exposure_vector
        )
    
    # Get residuals
    results = glm_nb.fit()
    spike_resid[f'neuron{neuron_id}'] = results.resid_response.values / exposure_vector
    spike_predict[f'neuron{neuron_id}'] = results.predict() / exposure_vector
    speed_coef[f'neuron{neuron_id}'] = results.params['speed']
    all_results[f'neuron{neuron_id}'] = results


# %% Plot
colors, dpi = figure_style()

f, axs = plt.subplots(2, 2, figsize=(1.75 * 2, 1.75 * 2), dpi=dpi)
axs = np.concatenate(axs)

axs[0].plot(rel_pos_list[PLOT_TRIAL], speed_pred[PLOT_TRIAL])
axs[0].set(ylabel='Speed (cm/s)', yticks=[0, 100, 200, 300, 400], yticklabels=[0, 10, 20, 30, 40],
           xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150])

for i in range(10):
    axs[1].plot(rel_pos_list[PLOT_TRIAL], spatial_basis[f'spatial_basis_{i}'][trial_id == PLOT_TRIAL])
axs[1].set(ylabel='Spatial weight', title='Position predictors',
           xticks=[0, 500, 1000, 1500], xticklabels=[0, 50, 100, 150],
           yticks=[0, 1], yticklabels=[0, 1])


axs[2].plot(rel_pos_list[PLOT_TRIAL],
            binned_spikes[f'neuron{PLOT_NEURON}'][PLOT_TRIAL] / time_per_bin_list[PLOT_TRIAL],
            label='Original')
axs[2].plot(rel_pos_list[PLOT_TRIAL], spike_predict[f'neuron{PLOT_NEURON}'][trial_id == PLOT_TRIAL],
            label='Prediction')
axs[2].set(xlabel='Position (cm)', ylabel='Firing rate (spks/s)', xticks=[0, 500, 1000, 1500],
           xticklabels=[0, 50, 100, 150])
axs[2].legend(bbox_to_anchor=(0.5, 0.9), prop={'size': 6})

axs[3].plot(rel_pos_list[PLOT_TRIAL], spike_resid[f'neuron{PLOT_NEURON}'][trial_id == PLOT_TRIAL])
axs[3].set(xlabel='Position (cm)', ylabel='Residual firing rate (spks/s)', xticks=[0, 500, 1000, 1500],
           xticklabels=[0, 50, 100, 150])

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'GLM_decoding_example.pdf')
