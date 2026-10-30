# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 10:10:28 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
from msvr_functions import paths, load_neural_data, load_subjects, load_objects, bin_signal

# Settings
BIN_SIZE = 0.05
MIN_TRIALS = 30
BEFORE_FIRST_OBJ_S = 3
N_LAGS = 3

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
    sniffing = np.load(session_path / 'continuous.breathing.npy')
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
    
    # This 'trials_data' DataFrame now contains all static info for each trial
    
    # --- Pre-filter Spike Times ---
    # Create a dictionary of spike times, keyed by neuron_id
    # This loop runs ONCE over neurons, not O(N_trials * N_neurons)
    print("Pre-filtering spike times by neuron...")
    spikes_by_neuron = {}
    for neuron_id in clusters['cluster_id']:
        # Get all spike times for this neuron and store them
        neuron_spikes = spikes['times'][np.isin(spikes['clusters'], neuron_id)]
        spikes_by_neuron[neuron_id] = neuron_spikes
    
    # Pre-initialize the output dictionary
    binned_spikes = {f'neuron{neuron_id}': [] for neuron_id in clusters['cluster_id']}
    
    ## 2. MAIN LOOP 
    
    print('Building predictor matrix...')
    reward_obj1, reward_obj2, no_rew_obj1, no_rew_obj2, at_obj3 = [], [], [], [], []
    expect_state, pos_pred, speed_pred, acc_pred, lick_pred, trial_id_list = [], [], [], [], [], []
    
    # Use itertuples() for a massive speedup over .index/.loc iteration
    # 'row' will be a namedtuple, access columns like row.enterEnvTime
    for row in trials_data.itertuples():
        
        if any(np.isnan([row.obj1_enter, row.obj2_enter, row.obj3_enter])):
            continue
        
        # Only one object should be rewarded
        if np.sum([row.obj1_rewarded, row.obj2_rewarded, row.obj3_rewarded]) != 1:
            continue
        
        # Get time bin edges for this trial
        bin_edges = np.arange(np.min([row.obj1_enter, row.obj2_enter, row.obj3_enter]) - BEFORE_FIRST_OBJ_S,
                              row.exitEnvTime,
                              step=BIN_SIZE)
        
        # Skip trials that are too short to have at least one full bin
        if len(bin_edges) < 2:
            continue
            
        n_bins = len(bin_edges) - 1
        bin_starts = bin_edges[:-1] # We only need the start of each bin for comparison
        trial_id_list.append(np.full(n_bins, row.Index))
    
        # --- Create object predictors ---
        # Access pre-processed data directly from 'row'
        # This is much clearer and faster (>= start, < end)
        reward_obj1.append(((bin_starts >= row.obj1_enter) & (bin_starts < row.obj1_exit)
                            & (row.obj1_rewarded == 1)).astype(int))
        no_rew_obj1.append(((bin_starts >= row.obj1_enter) & (bin_starts < row.obj1_exit)
                            & (row.obj1_rewarded == 0)).astype(int))
        reward_obj2.append(((bin_starts >= row.obj2_enter) & (bin_starts < row.obj2_exit)
                            & (row.obj2_rewarded == 1)).astype(int))
        no_rew_obj2.append(((bin_starts >= row.obj2_enter) & (bin_starts < row.obj2_exit)
                            & (row.obj2_rewarded == 0)).astype(int))
        at_obj3.append(((bin_starts >= row.obj3_enter) & (bin_starts < row.obj3_exit)).astype(int))
        
        # --- Create expectation predictor ---
        if (row.obj2_rewarded == 1) & (row.obj2_enter > row.obj3_enter):
            this_expect = ((bin_starts >= (row.obj3_exit))
                           & (bin_starts < row.obj2_enter)).astype(int)  
        elif (row.obj2_rewarded == 1) & (row.obj2_enter < row.obj3_enter):
            this_expect = ((bin_starts >= (row.obj1_exit))
                           & (bin_starts < row.obj2_enter)).astype(int)
        elif row.obj2_rewarded == 0:
            this_expect = np.zeros(n_bins, dtype=int)
        expect_state.append(this_expect)
        
        # --- Create other predictors ---
        pos_pred.append(bin_signal(timestamps, position - row.enterEnvPos, bin_edges))
        speed_pred.append(bin_signal(timestamps, running_speed, bin_edges))
        acc_pred.append(bin_signal(timestamps, acceleration, bin_edges))
        this_lick, _ = np.histogram(lick_times, bins=bin_edges) 
        lick_pred.append(this_lick)
            
        # --- Get binned spiking activity per neuron ---
        # Loop over the PRE-FILTERED dictionary
        for neuron_id, spike_times in spikes_by_neuron.items():
            
            # Get binned spikes
            these_spikes, _ = np.histogram(spike_times, bins=bin_edges)
            
            # Append to the pre-initialized list
            binned_spikes[f'neuron{neuron_id}'].append(these_spikes)
    
    print('Matrix build complete.')
    
    # --- Final Step (Post-processing) ---
    # You would now concatenate the lists into your final design matrix
    # This is much faster to do once at the end.
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
    
    # Add spike history 
    neuron_columns = [col for col in df.columns if col.startswith('neuron')]
    new_history_columns = []
    new_column_names = []
    
    for neuron_col in neuron_columns:
        for k in range(1, N_LAGS + 1):
            history_col_name = f"{neuron_col}_lag{k}"
            new_column_names.append(history_col_name)
            
            # Create the new column (Series)
            history_series = df.groupby('trial_id')[neuron_col].shift(k).fillna(0)
            
            # Give it the correct name
            history_series.name = history_col_name
            
            # Add the Series object to our list
            new_history_columns.append(history_series)

    # 2. Concatenate ALL new columns at once
    df = pd.concat([df] + new_history_columns, axis=1)
    
    # Identify predictor and response columns
    all_columns = df.columns
    neuron_columns = [col for col in all_columns if col.startswith('neuron') & (col[-4:-1] != 'lag')]
    predictor_columns = [col for col in all_columns if not col.startswith('neuron')]
    
    # 1. Identify the *base* predictors (non-neuron, non-trial_id)
    all_columns = df.columns
    base_predictor_columns = [
        col for col in all_columns 
        if not col.startswith('neuron') 
        and not col.startswith('trial_id')
        and not col.endswith(tuple(f'_lag{k}' for k in range(1, N_LAGS + 1)))
    ]
    
    # 2. Get the base X matrix and add the constant
    # We will add the history columns inside the loop
    X_base = df[base_predictor_columns]
    X_base_with_const = sm.add_constant(X_base, prepend=False)
    
    # 3. Loop, build neuron-specific X, fit, and store
    print(f"Fitting GLM with spike history for {len(neuron_columns)} neurons...")
    glm_dict = {}
    
    for neuron_col in neuron_columns:
        
        # Define the response variable 'y'
        y = df[neuron_col]
        
        # --- Define the neuron-specific X matrix ---
        
        # 1. Find the history columns *for this neuron only*
        this_neuron_history_cols = [
            f"{neuron_col}_lag{k}" for k in range(1, N_LAGS + 1)
        ]
        
        # 2. Combine the base predictors with this neuron's specific history
        X_neuron_specific = pd.concat(
            [X_base_with_const, df[this_neuron_history_cols]], 
            axis=1
        )
        
        # --- Fit the model ---
        glm_nb = sm.GLM(y, X_neuron_specific, 
                        family=sm.families.Poisson())
        
        try:
            results = glm_nb.fit()
            glm_dict[neuron_col] = results.pvalues[base_predictor_columns]
            
        except Exception as e:
            # Catch convergence errors, which are more common now
            print(f"Failed to fit model for {neuron_col}: {e}")
            glm_dict[neuron_col] = None
         
    print("GLM fitting complete.")
    
    # Convert to dataframe
    glm_df = pd.DataFrame(glm_dict).T.astype(float)
    
    # Drop neurons for which the fit failed
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
glm_results.to_csv(path_dict['save_path'] / 'GLM_results.csv', index=False)
    
    