# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:09:35 2025

By Guido Meijer & Gemini 2.5 Pro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pickle
from scipy.stats import zscore
from pathlib import Path
from msvr_functions import (paths, load_multiple_probes, calculate_peths, load_trials, figure_style,
                            peri_event_trace, load_objects)
colors, dpi = figure_style()

# Settings
subject = '459601'
date = '20240411'
BIN_SIZE = 25  # mm
SMOOTHING = 15

# Get paths
path_dict = paths()
session_path = Path(path_dict['local_data_path']) / 'Subjects' / subject / date

# Load in data
spikes, clusters, channels = load_multiple_probes(session_path)
trials = load_trials(subject, date)
obj_df = load_objects(subject, date)

# Loop over probes
for p, probe in enumerate(spikes.keys()):
    
    # Loop over regions
    for r, region in enumerate(np.unique(clusters[probe]['region'])):
        if region == 'root':
            continue
        print(f'Region {region}')
    
        # Get neurons in this region
        region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
        
        # Create binned spike matrix of entire task period
        peths, spike_matrix = calculate_peths(spikes[probe]['distances'], spikes[probe]['clusters'],
                                               region_neurons, [trials['enterEnvPos'].values[0]],
                                               pre_time=0, post_time=trials['enterEnvPos'].values[-1],
                                               bin_size=BIN_SIZE, smoothing=SMOOTHING)
        spike_matrix = np.squeeze(spike_matrix)  # (neurons x timebins)
        n_neurons, n_timebins = spike_matrix.shape
        time_ax = peths['tscale'] + trials['enterEnvTime'].values[0]

        # Drop neurons that don't have spikes
        spike_matrix = spike_matrix[np.sum(spike_matrix, axis=1) > 0, :]
                
        # 1. Z-score the spike matrix per neuron (already efficient)
        spikemat_zscore = zscore(spike_matrix, axis=1)
        
        # 2. Make correlation matrix 
        # np.corrcoef is a highly optimized way to compute the correlation matrix.
        # It expects variables as rows, which matches the input shape.
        corr_mat = np.corrcoef(spikemat_zscore)
        
        # 3. Get the eigenvalues and eigenvectors 
        # Use np.linalg.eigh as the correlation matrix is symmetric.
        eigenvalues, eigenvectors = np.linalg.eigh(corr_mat)
        
        # 4. Sort eigenvalues and eigenvectors in descending order 
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 5. Determine number of assemblies using Marchenko-Pastur law
        q = n_timebins / n_neurons if n_neurons > 0 else 0
        if q == 0:
            upper_bound = np.inf
        else:
            upper_bound = (1 + np.sqrt(1 / q))**2
        n_assemblies = np.sum(eigenvalues > upper_bound)
                
        # 6. Calculate assembly activation strength 
        # Isolate the significant eigenvectors (principal components)
        significant_pcs = eigenvectors[:, :n_assemblies]
        
        # The original calculation is a quadratic form: R_t = z_t^T * P_a * z_t
        # where P_a is the outer product of the eigenvector with its diagonal zeroed.
        # This can be vectorized across all timebins and assemblies.
        
        # Term 1: Projection of activity onto each principal component, squared.
        # Shape: (n_assemblies, n_timebins)
        projections = significant_pcs.T @ spikemat_zscore
        activation_main_term = projections**2
        
        # Term 2: Correction due to zeroing the diagonal of the outer product.
        # This is also vectorized using matrix multiplication.
        # Shape: (n_assemblies, n_timebins)
        correction_term = (significant_pcs**2).T @ (spikemat_zscore**2)
        
        # Final assembly activation is the difference between the two terms
        all_assembly_act_matrix = activation_main_term - correction_term
        
        # Return a list of arrays, matching the original code's output format
        all_assembly_act = list(all_assembly_act_matrix)
        
        # Plot assembly activity 
        for ia in range(n_assemblies):
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.75*3, 2.5), dpi=dpi)
            peri_event_trace(all_assembly_act[ia], time_ax,
                             obj_df.loc[obj_df['object'] == 1, 'distances'].values,
                             1 - obj_df.loc[obj_df['object'] == 1, 'goal'].values, ax=ax1, 
                             t_before=100, t_after=300, event_labels=['Goal', 'No goal'],
                             color_palette=[colors['goal'], colors['no-goal']])
            ax1.set(ylabel='Assembly activity', xlabel='Time from object entry (s)', title='Object 1',
                    xticks=[-100, 0, 100, 200, 300])
            
            peri_event_trace(all_assembly_act[ia], time_ax,
                             obj_df.loc[obj_df['object'] == 2, 'distances'].values,
                             1 - obj_df.loc[obj_df['object'] == 2, 'goal'].values, ax=ax2, 
                             t_before=100, t_after=300, event_labels=['Goal', 'No goal'],
                             color_palette=[colors['goal'], colors['no-goal']])
            ax2.set(ylabel='Assembly activity', xlabel='Time from object entry (s)', title='Object 2',
                    xticks=[-100, 0, 100, 200, 300])
            
            peri_event_trace(all_assembly_act[ia], time_ax,
                             obj_df.loc[obj_df['object'] == 3, 'distances'].values,
                             obj_df.loc[obj_df['object'] == 3, 'sound'].values, ax=ax3, 
                             t_before=100, t_after=300, event_labels=['Sound 1', 'Sound 2'],
                             color_palette=[colors['sound1'], colors['sound2']])
            ax3.set(ylabel='Assembly activity', xlabel='Time from object entry (s)', title='Object 3',
                    xticks=[-100, 0, 100, 200, 300])
            
            plt.suptitle(f'{region}')
            sns.despine(trim=True)
            plt.tight_layout()
            plt.show()
            
            f, ax1 = plt.subplots(1, 1, figsize=(4, 2), dpi=dpi)
            peri_event_trace(all_assembly_act[ia], time_ax,
                             trials['enterEnvPos'].values,
                             trials['soundId'].values, ax=ax1, 
                             t_before=0, t_after=1500, event_labels=['Sound 1', 'Sound 2'],
                             color_palette=[colors['sound1'], colors['sound2']])
            ax1.set(ylabel='Assembly activity', xlabel='Position in environment (mm)')
            
            sns.despine(trim=True)
            plt.tight_layout()
            plt.show()
            
        
        

