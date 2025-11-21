# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from msvr_functions import paths, load_neural_data, load_subjects, load_objects

# Settings
BIN_SIZE = 300  # mm
MIN_TRIALS = 30
MIN_SPEED = 20 # mm/s
N_LAGS = 3 # Spatial lags

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
scaler = StandardScaler()

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
    lick_times = np.load(session_path / 'lick.times.npy')
    all_obj_df = load_objects(subject, date)
        
    # Get the acceleration as the first derivative of the speed
    acceleration = np.gradient(running_speed)
        
    # Pivot the data to get enter/exit/reward status for each trial
    # This aligns all object data with the trial index
    obj_enter_times = all_obj_df.pivot(index='trial_nr', columns='object', values='times')
    obj_exit_times = all_obj_df.pivot(index='trial_nr', columns='object', values='exit_times')
    obj_rewarded = all_obj_df.pivot(index='trial_nr', columns='object', values='rewarded')
    
    # Rename columns for easy access
    col_mapping = {1: 'obj1', 2: 'obj2', 3: 'obj3'}
    trials_data = trials.join(obj_enter_times.rename(columns=col_mapping).add_suffix('_enter'))
    trials_data = trials_data.join(obj_exit_times.rename(columns=col_mapping).add_suffix('_exit'))
    trials_data = trials_data.join(obj_rewarded.rename(columns=col_mapping).add_suffix('_rewarded'))
    
    # Interpolate position to get position at specific times
    # We need this to map object entry/exit times to positions
    f_pos = interp1d(timestamps, position, fill_value="extrapolate")
    
    # Interpolate speed and position to spike times (if not already present, but let's do it to be safe/consistent)
    # Assuming spikes['times'] exists. 
    # Note: load_neural_data might add 'distances' and 'speeds' but let's calculate them to ensure they match our 'position' and 'running_speed' arrays
    spike_pos = f_pos(spikes['times'])
    f_speed = interp1d(timestamps, running_speed, fill_value="extrapolate")
    spike_speed = f_speed(spikes['times'])
    
    # Filter spikes by speed
    valid_spike_mask = spike_speed > MIN_SPEED
    
    # Create a dictionary of valid spike POSITIONS, keyed by neuron_id
    print("Pre-filtering spike times by neuron and speed...")
    spikes_by_neuron = {}
    for neuron_id in clusters['cluster_id']:
        # Get spikes for this neuron that also pass the speed threshold
        neuron_mask = np.isin(spikes['clusters'], neuron_id) & valid_spike_mask
        spikes_by_neuron[neuron_id] = spike_pos[neuron_mask]
    
    # Pre-initialize the output dictionary
    binned_spikes = {f'neuron{neuron_id}': [] for neuron_id in clusters['cluster_id']}
    
    ## 2. MAIN LOOP 
    
    print('Building predictor matrix...')
    reward_obj1, reward_obj2, no_rew_obj1, no_rew_obj2, at_obj3 = [], [], [], [], []
    expect_state, pos_pred, speed_pred, acc_pred, lick_pred, trial_id_list = [], [], [], [], [], []
    
    for row in trials_data.itertuples():
        
        if any(np.isnan([row.obj1_enter, row.obj2_enter, row.obj3_enter])):
            continue
        
        # Only one object should be rewarded
        if np.sum([row.obj1_rewarded, row.obj2_rewarded, row.obj3_rewarded]) != 1:
            continue
            
        # Get trial start and end positions
        start_pos = row.enterEnvPos
        end_pos = row.exitEnvPos
        
        # Define spatial bin edges for this trial (relative to start_pos for binning, but we use absolute for filtering)
        # Actually, let's work in trial-relative coordinates for the bins to be consistent
        trial_len = end_pos - start_pos
        if trial_len <= 0: continue
            
        bin_edges_rel = np.arange(0, trial_len, step=BIN_SIZE)
        
        # Skip trials that are too short
        if len(bin_edges_rel) < 2:
            continue
            
        bin_edges_abs = bin_edges_rel + start_pos
        n_bins = len(bin_edges_rel) - 1
        bin_centers_rel = bin_edges_rel[:-1] + BIN_SIZE/2
        bin_starts_rel = bin_edges_rel[:-1]
        
        trial_id_list.append(np.full(n_bins, row.Index))
        
        # --- Get Object Positions ---
        # Map time events to positions
        obj1_enter_pos = f_pos(row.obj1_enter) - start_pos
        obj1_exit_pos = f_pos(row.obj1_exit) - start_pos
        obj2_enter_pos = f_pos(row.obj2_enter) - start_pos
        obj2_exit_pos = f_pos(row.obj2_exit) - start_pos
        obj3_enter_pos = f_pos(row.obj3_enter) - start_pos
        obj3_exit_pos = f_pos(row.obj3_exit) - start_pos
        
        # --- Create object predictors (Spatial) ---
        reward_obj1.append(((bin_starts_rel >= obj1_enter_pos) & (bin_starts_rel < obj1_exit_pos)
                            & (row.obj1_rewarded == 1)).astype(int))
        no_rew_obj1.append(((bin_starts_rel >= obj1_enter_pos) & (bin_starts_rel < obj1_exit_pos)
                            & (row.obj1_rewarded == 0)).astype(int))
        reward_obj2.append(((bin_starts_rel >= obj2_enter_pos) & (bin_starts_rel < obj2_exit_pos)
                            & (row.obj2_rewarded == 1)).astype(int))
        no_rew_obj2.append(((bin_starts_rel >= obj2_enter_pos) & (bin_starts_rel < obj2_exit_pos)
                            & (row.obj2_rewarded == 0)).astype(int))
        at_obj3.append(((bin_starts_rel >= obj3_enter_pos) & (bin_starts_rel < obj3_exit_pos)).astype(int))
        
        # --- Create expectation predictor (Spatial) ---
        # Logic: After previous object exit until next object enter
        if (row.obj2_rewarded == 1) & (obj2_enter_pos > obj3_enter_pos):
            this_expect = ((bin_starts_rel >= (obj3_exit_pos))
                           & (bin_starts_rel < obj2_enter_pos)).astype(int)  
        elif (row.obj2_rewarded == 1) & (obj2_enter_pos < obj3_enter_pos):
            this_expect = ((bin_starts_rel >= (obj1_exit_pos))
                           & (bin_starts_rel < obj2_enter_pos)).astype(int)
        elif row.obj2_rewarded == 0:
            this_expect = np.zeros(n_bins, dtype=int)
        expect_state.append(this_expect)
        
        # --- Filter Continuous Data by Speed and Trial ---
        # Get indices for this trial
        trial_mask = (timestamps >= row.enterEnvTime) & (timestamps <= row.exitEnvTime)
        
        # Get data for this trial
        trial_pos = position[trial_mask]
        trial_speed = running_speed[trial_mask]
        trial_acc = acceleration[trial_mask]
        
        # Apply speed threshold
        speed_mask = trial_speed > MIN_SPEED
        
        valid_pos = trial_pos[speed_mask]
        valid_speed = trial_speed[speed_mask]
        valid_acc = trial_acc[speed_mask]
        
        # If no data passes threshold, fill with zeros (or nans, but zeros is safer for GLM)
        # However, if we have empty bins, binned_statistic returns NaN or 0.
        
        # Bin continuous signals
        # We bin 'valid_pos' into 'bin_edges_abs'.
        # Note: valid_pos are absolute positions. bin_edges_abs are absolute.
        
        # Position predictor (bin centers relative to start)
        pos_pred.append(bin_centers_rel)
        
        # Speed predictor
        if len(valid_pos) > 0:
            this_speed, _, _ = binned_statistic(valid_pos, valid_speed, statistic='mean', bins=bin_edges_abs)
            this_acc, _, _ = binned_statistic(valid_pos, valid_acc, statistic='mean', bins=bin_edges_abs)
        else:
            this_speed = np.zeros(n_bins)
            this_acc = np.zeros(n_bins)
            
        # Fill NaNs (bins with no high-speed data) with 0 or mean? 
        # If a bin has NO high speed data, it shouldn't really contribute, but we are keeping the bin structure fixed.
        # Let's fill with 0 for now, or maybe the global mean? 0 is safer for "no speed".
        this_speed = np.nan_to_num(this_speed)
        this_acc = np.nan_to_num(this_acc)
        
        speed_pred.append(this_speed)
        acc_pred.append(this_acc)
        
        # Licks
        # Get licks in this trial
        trial_lick_mask = (lick_times >= row.enterEnvTime) & (lick_times <= row.exitEnvTime)
        trial_lick_times = lick_times[trial_lick_mask]
        
        # Get lick positions
        trial_lick_pos = f_pos(trial_lick_times)
        trial_lick_speed = f_speed(trial_lick_times)
        
        # Filter licks by speed
        valid_lick_pos = trial_lick_pos[trial_lick_speed > MIN_SPEED]
        
        this_lick, _ = np.histogram(valid_lick_pos, bins=bin_edges_abs)
        lick_pred.append(this_lick)
            
        # --- Get binned spiking activity per neuron ---
        for neuron_id, neuron_spike_pos in spikes_by_neuron.items():
            
            # Filter spikes for this trial (by position range)
            # They are already filtered by speed globally
            
            # We need spikes that are within [start_pos, end_pos]
            # Note: neuron_spike_pos are absolute positions
            trial_spike_mask = (neuron_spike_pos >= start_pos) & (neuron_spike_pos <= end_pos)
            these_spike_pos = neuron_spike_pos[trial_spike_mask]
            
            # Bin spikes
            these_spikes, _ = np.histogram(these_spike_pos, bins=bin_edges_abs)
            
            # Append
            binned_spikes[f'neuron{neuron_id}'].append(these_spikes)
    
    print('Matrix build complete.')
    
    # --- Final Step (Post-processing) ---
    final_predictors = {
        'reward_at_object1': np.concatenate(reward_obj1),
        'no_reward_at_object1': np.concatenate(no_rew_obj1),
        'reward_at_object2': np.concatenate(reward_obj2),
        'no_reward_at_object2': np.concatenate(no_rew_obj2),
        'at_object3': np.concatenate(at_obj3),
        'expect_state': np.concatenate(expect_state),
        'position': np.concatenate(pos_pred), 
        'speed': np.concatenate(speed_pred),  
        'acceleration': np.concatenate(acc_pred),
        'lick': np.concatenate(lick_pred),
        'trial_id': np.concatenate(trial_id_list)
    }
    
    # Add neural data
    for neuron_key, spike_list in binned_spikes.items():
        final_predictors[neuron_key] = np.concatenate(spike_list)
    
    # Convert to pandas dataframe
    df = pd.DataFrame(final_predictors)
    
    # Add spike history (Spatial Lags)
    neuron_columns = [col for col in df.columns if col.startswith('neuron')]
    new_history_columns = []
    
    for neuron_col in neuron_columns:
        for k in range(1, N_LAGS + 1):
            history_col_name = f"{neuron_col}_lag{k}"
            
            # Create the new column (Series)
            history_series = df.groupby('trial_id')[neuron_col].shift(k).fillna(0)
            history_series.name = history_col_name
            new_history_columns.append(history_series)

    # Concatenate ALL new columns at once
    df = pd.concat([df] + new_history_columns, axis=1)
    
    # Identify predictor and response columns
    all_columns = df.columns
    base_predictor_columns = [
        col for col in all_columns 
        if not col.startswith('neuron') 
        and not col.startswith('trial_id')
        and not col.endswith(tuple(f'_lag{k}' for k in range(1, N_LAGS + 1)))
    ]
    
    # Get the base X matrix and add the constant
    X_base = df[base_predictor_columns]
    X_base_with_const = sm.add_constant(X_base, prepend=False)
    
    # Fit GLM
    print(f"Fitting GLM with spatial spike history for {len(neuron_columns)} neurons...")
    glm_dict = {}
    
    for neuron_col in neuron_columns:
        
        y = df[neuron_col]
        
        this_neuron_history_cols = [
            f"{neuron_col}_lag{k}" for k in range(1, N_LAGS + 1)
        ]
        
        X_neuron_specific = pd.concat(
            [X_base_with_const, df[this_neuron_history_cols]], 
            axis=1
        )
        
        glm_nb = sm.GLM(y, X_neuron_specific, 
                        family=sm.families.Poisson())
        
        try:
            results = glm_nb.fit()
            glm_dict[neuron_col] = results.pvalues[base_predictor_columns]
            
        except Exception as e:
            print(f"Failed to fit model for {neuron_col}: {e}")
            glm_dict[neuron_col] = None
         
    print("GLM fitting complete.")
    
    # Convert to dataframe
    glm_df = pd.DataFrame(glm_dict).T.astype(float)
    glm_df = glm_df.dropna(how='all')
    
    # Add neuron level info
    glm_df['neuron_id'] = np.array([i[6:] for i in glm_df.index]).astype(int)
    glm_df = glm_df.reset_index(drop=True)
    glm_df['acronym'] = clusters['acronym'][np.isin(clusters['cluster_id'], glm_df['neuron_id'])]
    glm_df['region'] = clusters['region'][np.isin(clusters['cluster_id'], glm_df['neuron_id'])]
    
    # Add session level info
    glm_df['subject'] = subject
    glm_df['date'] = date
    glm_df['probe'] = probe
    
    # Add to overal dataframe
    glm_results = pd.concat((glm_results, glm_df))
    
# Save to disk
glm_results.to_csv(path_dict['save_path'] / 'GLM_results_position.csv', index=False)
