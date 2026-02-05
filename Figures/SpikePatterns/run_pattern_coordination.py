# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:18:07 2026

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_objects, figure_style
colors, dpi = figure_style()

MIN_RIPPLES = 10
PLOT = False

# Initialize
path_dict = paths()
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)

# Functions
def get_event_triggered_activity(amplitudes, times, event_times, window_sec=(-2, 2)):
    """
    Slices continuous amplitudes into event-centric windows.
    Returns: (n_events, n_patterns, n_bins)
    """
    dt = np.mean(np.diff(times))
    # Convert window seconds to bins
    bins_pre = int(abs(window_sec[0]) / dt)
    bins_post = int(abs(window_sec[1]) / dt)
    
    n_events = len(event_times)
    n_patterns = amplitudes.shape[0]
    n_bins = bins_pre + bins_post
    
    tensor = np.zeros((n_events, n_patterns, n_bins))
    
    for i, evt in enumerate(event_times):
        # Find index closest to event time
        idx = np.searchsorted(times, evt)
        
        # Safety check for boundaries
        start = idx - bins_pre
        stop = idx + bins_post
        
        if start >= 0 and stop < amplitudes.shape[1]:
            tensor[i, :, :] = amplitudes[:, start:stop]
        else:
            tensor[i, :, :] = np.nan # Handle edge cases
            
    return tensor, np.linspace(window_sec[0], window_sec[1], n_bins)

def compute_noise_correlations(tensor_A, tensor_B):
    """
    Computes time-resolved correlations between all pattern pairs 
    across trials, removing the mean event response (PSTH).
    
    tensor_A: (n_events, n_patterns_A, n_bins)
    tensor_B: (n_events, n_patterns_B, n_bins)
    
    Returns: Correlation matrix (n_patterns_A, n_patterns_B, n_bins)
    """
    # 1. Compute the PSTH (Mean across trials)
    psth_A = np.nanmean(tensor_A, axis=0)
    psth_B = np.nanmean(tensor_B, axis=0)
    
    # 2. Subtract PSTH to get residuals (Single trial deviations)
    # Broadcasting: (Events, Patterns, Bins) - (Patterns, Bins)
    residuals_A = tensor_A - psth_A[None, :, :]
    residuals_B = tensor_B - psth_B[None, :, :]
    
    n_pat_A = tensor_A.shape[1]
    n_pat_B = tensor_B.shape[1]
    n_bins = tensor_A.shape[2]
    
    corr_matrix = np.zeros((n_pat_A, n_pat_B, n_bins))
    
    # 3. Correlate residuals at each time bin
    for t in range(n_bins):
        # Extract data for this time bin across all events
        # Shape: (n_events, n_patterns)
        data_A = residuals_A[:, :, t]
        data_B = residuals_B[:, :, t]
        
        # Calculate correlation matrix for this slice
        # Rowvar=False assumes columns are variables (patterns)
        # We need to manually handle cross-correlation between two distinct sets
        for i in range(n_pat_A):
            for j in range(n_pat_B):
                # Correlation of pattern i (Reg A) vs pattern j (Reg B) across trials
                valid = ~np.isnan(data_A[:, i]) & ~np.isnan(data_B[:, j])
                if np.sum(valid) > 2:
                    corr = np.corrcoef(data_A[valid, i], data_B[valid, j])[0, 1]
                    corr_matrix[i, j, t] = corr
                else:
                    corr_matrix[i, j, t] = np.nan
                    
    return corr_matrix

def visualize_coactivation(coactivation_map, time_axis, region_A_name, region_B_name, path_dict,
                           subject, date, event):
    """
    Plots the noise correlations between pattern pairs over time.
    """
    n_pat_A, n_pat_B, n_bins = coactivation_map.shape
    
    # --- 1. Flatten Data for the Heatmap ---
    # We want rows to be "Pattern Pairs" and columns to be "Time"
    reshaped_map = coactivation_map.reshape(n_pat_A * n_pat_B, n_bins)
    
    # Generate labels for the Y-axis (e.g., "PERI0-TEa1")
    pair_labels = []
    for i in range(n_pat_A):
        for j in range(n_pat_B):
            pair_labels.append(f"{region_A_name}{i} - {region_B_name}{j}")
            
    # Sort pairs by their peak correlation in the PRE-EVENT window
    # This filters out noise and highlights the relevant pairs
    #pre_event_idx = time_axis < 0
    peak_pre_corr = np.mean(reshaped_map, axis=1)
    sorted_indices = np.argsort(peak_pre_corr)[::-1] # Descending order
    
    sorted_map = reshaped_map[sorted_indices, :]
    sorted_labels = [pair_labels[i] for i in sorted_indices]

    # --- 2. Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    
    # Plot A: Heatmap of All Pairs
    sns.heatmap(sorted_map, ax=axes[0], center=0, cmap="vlag", 
                xticklabels=10, yticklabels=sorted_labels)
    
    # Fix X-axis labels to show time instead of bin index
    xticks = axes[0].get_xticks()
    # Map bin indices to time values
    xlabels = [f"{time_axis[int(x)]:.1f}" for x in xticks] 
    axes[0].set_xticklabels(xlabels, rotation=0)
    axes[0].set_title(f"Noise Correlation: {region_A_name} vs {region_B_name} (Sorted by Strength)")
    axes[0].axvline(x=np.searchsorted(time_axis, 0), color='k', linestyle='--', linewidth=2)
    axes[0].set_ylabel("Pattern Pairs")
    
    # Plot B: Trace of the Top Pair vs. Zero
    # Identify the strongest pair
    top_pair_idx = sorted_indices[0]
    top_pair_data = reshaped_map[top_pair_idx, :]
    top_label = pair_labels[top_pair_idx]
    
    axes[1].plot(time_axis, top_pair_data, label=f"Top Pair: {top_label}", color='#d62728', linewidth=2)
    axes[1].axhline(0, color='k', linewidth=1)
    axes[1].axvline(0, color='k', linestyle='--', label="Event Onset")
    
    # Style
    axes[1].set_xlabel("Time from Event (s)")
    axes[1].set_ylabel("Noise Correlation (r)")
    axes[1].set_title(f"Time Course of Dominant Co-activation: {top_label}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.savefig(path_dict['google_drive_fig_path'] / 'CoactivationPatterns' / f'{event}' 
                / f'{subject}_{date}_{region_A_name}-{region_B_name}.jpg')
    plt.close(fig)

# %% MAIN

coact_df, ripple_df = pd.DataFrame(), pd.DataFrame()
for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    
    # Load in data for this session
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
    trials = pd.read_csv(session_path / 'trials.csv')
    these_ripples = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
    obj_df = load_objects(subject, date)
    obj2_times = obj_df.loc[obj_df['object'] == 2, 'times'].values
    
    # Get paths to data of this session
    amp_paths = (path_dict['google_drive_data_path'] / 'SpikePatterns').glob(f'{subject}_{date}*.amplitudes.npy')
    
    # Loop over regions and get spike pattern amplitudes    
    amplitudes = dict()
    for amp_path in amp_paths:
        
        # Load in data for this region
        _, _, region = amp_path.stem.split('.')[0].split('_')
        amplitudes[region] = np.load(amp_path)
        times = np.load(amp_path.parent / (amp_path.stem.split('.')[0] + '.times.npy'))
        
    # Loop over region pairs and get co-activation
    these_regions = list(amplitudes.keys())
    for r1, region_A in enumerate(these_regions[:-1]):
        for r2, region_B in enumerate(these_regions[r1+1:]):
            
            # OBJECT 2
            # Extract event windows
            tensor_A, time_axis = get_event_triggered_activity(
                amplitudes[region_A], times, obj2_times, window_sec=(-2.0, 2.0)
            )
            tensor_B, _ = get_event_triggered_activity(
                amplitudes[region_B], times, obj2_times, window_sec=(-2.0, 2.0)
            )
            
            # Compute correlations
            coactivation_map = compute_noise_correlations(tensor_A, tensor_B)
            
            # Extract strongest co-activation pair (pre object entry)
            n_pat_A, n_pat_B, n_bins = coactivation_map.shape
            reshaped_map = coactivation_map.reshape(n_pat_A * n_pat_B, n_bins)
            pre_event_idx = time_axis < 0
            peak_corr = np.mean(reshaped_map, axis=1)
            top_pair_data = reshaped_map[np.argmax(peak_corr), :]
            baseline_sub = top_pair_data - np.mean(top_pair_data[(time_axis >= -1.5) & (time_axis < -0.5)])
            
            # Add to df
            coact_df = pd.concat((coact_df, pd.DataFrame(data={
                'co-activation': top_pair_data, 'baseline_subracted': baseline_sub,
                'time': time_axis,
                'region_A': region_A, 'region_B': region_B, 'region_pair': f'{region_A}-{region_B}',
                'subject': subject, 'date': date
                })))
            
            if PLOT:
                # Plot 
                visualize_coactivation(coactivation_map, time_axis, region_A, region_B, path_dict,
                                       subject, date, 'Obj2')
            
            # RIPPLES
            if these_ripples.shape[0] < MIN_RIPPLES:
                continue
            
            # Extract event windows
            tensor_A, time_axis = get_event_triggered_activity(
                amplitudes[region_A], times, these_ripples['start_times'], window_sec=(-2, 2)
            )
            tensor_B, _ = get_event_triggered_activity(
                amplitudes[region_B], times, these_ripples['start_times'], window_sec=(-2, 2)
            )
            
            # Compute correlations
            coactivation_map = compute_noise_correlations(tensor_A, tensor_B)
            
            # Extract strongest co-activation pair (pre object entry)
            n_pat_A, n_pat_B, n_bins = coactivation_map.shape
            reshaped_map = coactivation_map.reshape(n_pat_A * n_pat_B, n_bins)
            peak_corr = np.mean(reshaped_map, axis=1)
            top_pair_data = reshaped_map[np.argmax(peak_corr), :]
            baseline_sub = top_pair_data - np.mean(top_pair_data[(time_axis >= -1.5) & (time_axis < -0.5)])
            
            # Add to df
            ripple_df = pd.concat((ripple_df, pd.DataFrame(data={
                'co-activation': top_pair_data, 'baseline_subracted': baseline_sub,
                'time': time_axis,
                'region_A': region_A, 'region_B': region_B, 'region_pair': f'{region_A}-{region_B}',
                'subject': subject, 'date': date
                })))
            
            if PLOT:
                # Plot 
                visualize_coactivation(coactivation_map, time_axis, region_A, region_B, path_dict,
                                       subject, date, 'Ripples')
            
            
            
# %% Plot

f, axs = plt.subplots(3, 5, figsize=(1.75 *3, 1.75 * 5), dpi=dpi, sharey=True, sharex=True)
axs = axs.flatten()
for i, region_pair in enumerate(coact_df['region_pair'].unique()):
    sns.lineplot(data=coact_df[coact_df['region_pair'] == region_pair], x='time', y='baseline_subracted',
                 ax=axs[i], errorbar='se', err_kws={'lw': 0})
    axs[i].set(title=region_pair)
sns.despine(trim=True)
plt.tight_layout()

# %%
f, axs = plt.subplots(3, 5, figsize=(7, 4), dpi=dpi, sharey=False, sharex=True)
axs = axs.flatten()
for i, region_pair in enumerate(ripple_df['region_pair'].unique()):
    axs[i].plot([-2, 2], [0, 0], ls='--', lw=0.5, color='grey')
    sns.lineplot(data=ripple_df[ripple_df['region_pair'] == region_pair], x='time', y='baseline_subracted',
                 ax=axs[i], errorbar='se', err_kws={'lw': 0})
    axs[i].set(title=region_pair, ylabel='', xlabel='')
sns.despine(trim=True)
plt.tight_layout()
    