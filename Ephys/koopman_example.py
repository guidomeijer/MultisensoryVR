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
REGION_A = 'CA1'
REGION_B = 'PERI 36'

# Session selection
subject = '466394'
date = '20241108'
probe = 'probe00'

# Initialize
path_dict = paths(sync=False)

# Load in data
session_path = path_dict['local_data_path'] / 'subjects' / f'{subject}' / f'{date}'
trials = pd.read_csv(session_path / 'trials.csv')
spikes, clusters, channels = load_neural_data(session_path, probe)
all_obj_df = load_objects(subject, date)

# Construct trial list

for i, trial_time in enumerate(all_obj_df.loc[all_obj_df['object'] == 2, 'times']):
    
    these_spike_times, these_spike_ids = [], []
    asd
    #spikes['times'][np.isin(spikes['clusters'], clusters['region'])
    
    





# ==========================================
# Example Usage with Multi-Trial Data
# ==========================================

if __name__ == "__main__":
    print("Generating simulated Multi-Trial data...")
    n_trials = 20
    trial_duration = 2.0 # seconds (window of interest)
    n_neurons_A = 20
    n_neurons_B = 20
    
    # List to store (times, ids) for each trial
    all_trials = []
    
    # Simulate trials where Area A drives Area B 
    for i in range(n_trials):
        t_steps = np.arange(0, trial_duration, 0.001)
        
        # Variable rhythm per trial (non-stationary across trials, but we fit average K)
        freq = 10 + np.random.normal(0, 1) # ~10Hz
        rhythm = np.sin(2 * np.pi * freq * t_steps)
        rhythm[rhythm < 0] = 0
        
        trial_times = []
        trial_ids = []
        
        # Area A
        for n in range(n_neurons_A):
            rate = rhythm * 30 
            is_spike = np.random.rand(len(t_steps)) < (rate * 0.001)
            spikes = t_steps[is_spike]
            trial_times.extend(spikes)
            trial_ids.extend([n] * len(spikes))
            
        # Area B (Delayed copy of A)
        delay_steps = 30 # 30ms delay
        rhythm_B = np.roll(rhythm, delay_steps)
        for n in range(n_neurons_B):
            rate = rhythm_B * 30
            is_spike = np.random.rand(len(t_steps)) < (rate * 0.001)
            spikes = t_steps[is_spike]
            trial_times.extend(spikes)
            trial_ids.extend([n + n_neurons_A] * len(spikes))
            
        all_trials.append((np.array(trial_times), np.array(trial_ids)))

    area_map = {
        'A': list(range(n_neurons_A)),
        'B': list(range(n_neurons_A, n_neurons_A + n_neurons_B))
    }
    
    # Run Pipeline
    pipeline = NeuralKoopmanPipeline(dt=0.01, sigma=0.02, n_delays=10)
    
    # 1. Preprocess all trials
    trials_X_A, trials_X_B = pipeline.preprocess_trials(all_trials, area_map)
    
    # 2. Fit (automatically stacks Hankel matrices)
    pipeline.fit_koopman_operators(trials_X_A, trials_X_B)
    
    # 3. Analyze
    c_AB, c_BA = pipeline.analyze_causality(trials_X_A, trials_X_B)
    
    print("\n--- Multi-Trial Results ---")
    print(f"Causality Area A -> Area B: {c_AB:.4f}")
    print(f"Causality Area B -> Area A: {c_BA:.4f}")
    
    pipeline.plot_eigenvalues()