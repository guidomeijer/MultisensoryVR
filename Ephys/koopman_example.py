# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
from msvr_functions import paths, load_neural_data, load_objects, bin_signal, figure_style
from koopman import NeuralKoopmanPipeline

# Settings
T_BEFORE = 2  # s
T_AFTER = 2  # s
REGION_A = 'AUD'
REGION_B = 'TEa'

# Session selection
subject = '466394'
date = '20241108'
probe = 'probe00'

# Initialize
path_dict = paths(sync=False)

# Load in data
session_path = path_dict['local_data_path'] / 'subjects' / f'{subject}' / f'{date}'
trials = pd.read_csv(session_path / 'trials.csv')
spikes, clusters, channels = load_neural_data(session_path, probe, min_fr=0.5)
all_obj_df = load_objects(subject, date)

# Get region spikes
region_a_neurons = clusters['cluster_id'][clusters['region'] == REGION_A]
region_a_spikes = spikes['times'][np.isin(spikes['clusters'], region_a_neurons)]
region_a_clusters = spikes['clusters'][np.isin(spikes['clusters'], region_a_neurons)]
region_b_neurons = clusters['cluster_id'][clusters['region'] == REGION_B]
region_b_spikes = spikes['times'][np.isin(spikes['clusters'], region_b_neurons)]
region_b_clusters = spikes['clusters'][np.isin(spikes['clusters'], region_b_neurons)]

# Construct trial list
all_trials = []
for i, trial_time in enumerate(all_obj_df.loc[all_obj_df['object'] == 2, 'times']):
    
    these_spike_times, these_spike_ids = [], []
    
    # Area A
    these_spike_times.extend(region_a_spikes[
        (region_a_spikes > (trial_time - T_BEFORE)) & (region_a_spikes < (trial_time + T_AFTER))
        ])
    these_spike_ids.extend(region_a_clusters[
        (region_a_spikes > (trial_time - T_BEFORE)) & (region_a_spikes < (trial_time + T_AFTER))
        ]) 
        
    # Area B
    these_spike_times.extend(region_b_spikes[
        (region_b_spikes > (trial_time - T_BEFORE)) & (region_b_spikes < (trial_time + T_AFTER))
        ])
    these_spike_ids.extend(region_b_clusters[
        (region_b_spikes > (trial_time - T_BEFORE)) & (region_b_spikes < (trial_time + T_AFTER))
        ]) 
    
    all_trials.append((np.array(these_spike_times), np.array(these_spike_ids), trial_time - T_BEFORE))
    
# Create neuron map
area_map = {
    'A': list(np.unique(region_a_clusters)),
    'B': list(np.unique(region_b_clusters))
}

# Run Pipeline
pipeline = NeuralKoopmanPipeline(dt=0.05, sigma=0.01, n_delays=10)

# 1. Preprocess all trials
trials_X_A, trials_X_B = pipeline.preprocess_trials(all_trials, area_map)

# Sanity Check: Ensure data isn't empty and shapes are correct
if len(trials_X_A) > 0:
    print(f"\n--- Data Verification ---")
    print(f"Successfully processed {len(trials_X_A)} trials.")
    print(f"Trial 0 Matrix Shape (Time x Neurons): {trials_X_A[0].shape}")
    
    # Check if we have enough time points for the delay embedding
    if trials_X_A[0].shape[0] <= pipeline.n_delays:
        raise ValueError(f"Trials are too short ({trials_X_A[0].shape[0]} bins) for {pipeline.n_delays} delays.")
else:
    raise ValueError("No trials were processed. Check your T_BEFORE/T_AFTER or object timestamps.")

# Fit the operators
# This vertically stacks the Hankel matrices of all trials to learn one common dynamical law
print("\nFitting Koopman Operators (this may take a moment)...")
pipeline.fit_koopman_operators(trials_X_A, trials_X_B)

# ---------------------------------------------------------
# 3. Causality Analysis
# ---------------------------------------------------------

print("Calculating Directional Influence...")
c_AB, c_BA = pipeline.analyze_causality(trials_X_A, trials_X_B)

# ---------------------------------------------------------
# 4. Output & Visualization
# ---------------------------------------------------------

print("\n" + "="*50)
print(f"INTER-AREA COMMUNICATION RESULTS: {REGION_A} <-> {REGION_B}")
print("="*50)
print(f"Causality {REGION_A} -> {REGION_B} : {c_AB:.4f}")
print(f"Causality {REGION_B} -> {REGION_A} : {c_BA:.4f}")

# Simple text interpretation
diff = c_AB - c_BA
if diff > 0.02: 
    print(f">> Conclusion: Dominant drive from {REGION_A} to {REGION_B}")
elif diff < -0.02:
    print(f">> Conclusion: Dominant drive from {REGION_B} to {REGION_A}")
else:
    print(f">> Conclusion: Balanced or Weak coupling")

# Plot A: Singular Values (Dynamical Modes)
# If these curves overlap significantly, it suggests shared oscillatory modes.
pipeline.plot_eigenvalues()

# Plot B: Summary Bar Chart
plt.figure(figsize=(6, 5))
sns.barplot(x=[f'{REGION_A} $\to$ {REGION_B}', f'{REGION_B} $\to$ {REGION_A}'], y=[c_AB, c_BA])
plt.ylabel('Koopman Prediction Improvement (Causality)')
plt.title(f'Directional Influence: {subject} ({date})')
plt.ylim(0, max(c_AB, c_BA) * 1.2) # Scale y-axis
sns.despine()
plt.show()

