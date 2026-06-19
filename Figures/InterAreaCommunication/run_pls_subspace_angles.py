# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:15:00 2026 by Guido Meijer
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSCanonical
from scipy.linalg import subspace_angles
from itertools import combinations
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
TIME_WIN = {'obj1': [-2, 2], 'obj2': [-2, 2], 'obj3': [-2, 2]}
BIN_SIZE = 0.025
SMOOTHING = 0.05
MIN_NEURONS = 5  # per region
N_COMPONENTS = 4
CORTICAL_REGIONS = ['VIS', 'PERI', 'TEa', 'AUD', 'LEC']
TARGET_REGION = 'CA1'

# Initialize
path_dict = paths()
subjects = load_subjects()

# Load in data
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv', dtype={'subject': str, 'date': str})
rec = rec.drop_duplicates(subset=['subject', 'date'])

def process_session(subject, date):
    print(f'Processing session {subject} {date}...')
    session_angles_df = pd.DataFrame()

    session_path = path_dict['local_data_path'] / 'Subjects' / str(subject) / str(date)
    spikes, clusters, channels = load_multiple_probes(session_path)
    all_obj_df = load_objects(subject, date)

    spikes_dict = {'obj1': dict(), 'obj2': dict(), 'obj3': dict()}
    
    for k, probe in enumerate(spikes.keys()):
        for j, region in enumerate(np.unique(clusters[probe]['region'])):
            if region == 'root':
                continue
            
            region_neurons = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]

            # First, calculate PETHs for both objects to determine active neurons
            binned_spikes_tmp = {}
            active_mask = np.zeros(len(region_neurons), dtype=bool)
            
            for m, obj in enumerate(['obj1', 'obj2', 'obj3']):
                peth, binned_spikes = calculate_peths(
                    spikes[probe]['times'], spikes[probe]['clusters'], region_neurons,
                    all_obj_df.loc[all_obj_df['object'] == m+1, 'times'].values,
                    np.abs(TIME_WIN[obj][0]), TIME_WIN[obj][1], BIN_SIZE, SMOOTHING, return_fr=False)
                
                binned_spikes_tmp[obj] = np.swapaxes(binned_spikes, 1, 2)
                active_mask = active_mask | (np.max(peth['means'], axis=1) > 0.01)
                
            if np.sum(active_mask) < MIN_NEURONS:
                continue
                
            for obj in ['obj1', 'obj2', 'obj3']:
                spikes_dropped = binned_spikes_tmp[obj][:, :, active_mask]
                spikes_dict[obj][region] = spikes_dropped
        
    # Check if CA1 is present for all objects
    if 'CA1' not in spikes_dict['obj1'] or 'CA1' not in spikes_dict['obj2'] or 'CA1' not in spikes_dict['obj3']:
        return session_angles_df

    available_cortical_regions = []
    for ctx_region in CORTICAL_REGIONS:
        if ctx_region in spikes_dict['obj1'] and ctx_region in spikes_dict['obj2'] and ctx_region in spikes_dict['obj3']:
            available_cortical_regions.append(ctx_region)

    if len(available_cortical_regions) < 2:
        return session_angles_df

    # Extract CA1 weight matrices for all available cortical regions
    weight_matrices = {}
    for ctx_region in available_cortical_regions:
        x_cortex = np.concatenate((spikes_dict['obj1'][ctx_region], spikes_dict['obj2'][ctx_region], spikes_dict['obj3'][ctx_region]), axis=0)
        y_ca1 = np.concatenate((spikes_dict['obj1']['CA1'], spikes_dict['obj2']['CA1'], spikes_dict['obj3']['CA1']), axis=0)
        
        n_trials, n_timebins, n_neurons_X = x_cortex.shape
        _, _, n_neurons_Y = y_ca1.shape

        X_2D = x_cortex.reshape(-1, n_neurons_X)
        Y_2D = y_ca1.reshape(-1, n_neurons_Y)
        
        # Fit PLS
        pls = PLSCanonical(n_components=N_COMPONENTS)
        pls.fit(X_2D, Y_2D)
        
        weight_matrices[ctx_region] = pls.y_weights_

    # Calculate subspace angles between all pairs of weight matrices
    rows = []
    for reg1, reg2 in combinations(available_cortical_regions, 2):
        if reg1 > reg2:  # make region pairs alphabetical
            reg1, reg2 = reg2, reg1
        w1 = weight_matrices[reg1]
        w2 = weight_matrices[reg2]
        
        # Compute subspace angles (sorted in descending order) and convert to degrees
        angles = np.rad2deg(subspace_angles(w1, w2))
        
        row = {
            'subject': subject,
            'date': date,
            'region1': reg1,
            'region2': reg2,
            'region_pair': f'{reg1}-{reg2}',
            'mean_angle': np.mean(angles),
            'median_angle': np.median(angles)
        }
        for i, angle in enumerate(angles):
            row[f'angle_{i+1}'] = angle
            
        rows.append(row)
        
    session_angles_df = pd.DataFrame(rows)
    return session_angles_df

if __name__ == '__main__':
    print(f'Processing {len(rec)} sessions sequentially...')
    
    results = []
    for _, row in rec.iterrows():
        df = process_session(row['subject'], row['date'])
        if not df.empty:
            results.append(df)

    if results:
        angles_df = pd.concat(results, ignore_index=True)
        out_path = path_dict['google_drive_data_path'] / 'cortex_ca1_pla_subspace_angles.csv'
        angles_df.to_csv(out_path, index=False)
        print(f'Processing complete. Saved results to {out_path}')
    else:
        print('No sessions had sufficient regions to calculate subspace angles.')
