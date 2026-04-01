# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import sem
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
T_BEFORE = 2
T_AFTER = 2
BIN_SIZE = 0.1
SMOOTHING = 0.12
SUBTRACT_MEAN = True
MIN_NEURONS = 5  # per region
N_CPUS = 18

# Initialize
path_dict = paths()
subjects = load_subjects()

# Load in data
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(subset=['subject', 'date'])
neurons_df = pd.read_csv(path_dict['save_path'] / 'significant_neurons.csv')
neurons_df['subject'] = neurons_df['subject'].astype(str)
neurons_df['date'] = neurons_df['date'].astype(str)

# Select neurons to include
neurons_df['include'] = (neurons_df['p_context_obj2'] < 0.05) | (neurons_df['p_obj_onset'] < 0.05) | (neurons_df['p_reward'] < 0.05)

# %% Function for parallelization

def run_mi(x, y):
    """
    Calculate average mutual information between all pairs of neurons in two regions.
    x: (trials x neurons_1) for a single time bin
    y: (trials x neurons_2) for a single time bin
    """
    # Calculate pairwise mutual information between all neuron combinations
    pairwise_mi = []
    for j in range(y.shape[1]):
        # mutual_info_regression calculates MI between each column in x and the vector y[:, j]
        mi_scores = mutual_info_regression(x, y[:, j])
        pairwise_mi.extend(mi_scores)

    if not pairwise_mi:
        return np.nan, np.nan

    mi_mean = np.nanmean(pairwise_mi)
    mi_sem = sem(pairwise_mi, nan_policy='omit')
    return mi_mean, mi_sem

# %%
mi_results = []

with Parallel(n_jobs=N_CPUS) as parallel:
    for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
        print(f'Recording {i} of {len(rec)}: \n{subject} {date}')

        # Load in neural data for all probes
        session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
        spikes, clusters, channels = load_multiple_probes(session_path)
        all_obj_df = load_objects(subject, date)
        session_neurons = neurons_df[(neurons_df['subject'] == subject) & (neurons_df['date'] == date)]

        # Initialize containers for this session
        obj_data = {'obj1_rew': {}, 'obj1_no_rew': {}, 'obj2_rew': {}, 'obj2_no_rew': {}}
        cond_masks = [
            ('obj1_rew', (all_obj_df['goal'] == 1) & (all_obj_df['object'] == 1)),
            ('obj1_no_rew', (all_obj_df['goal'] == 0) & (all_obj_df['object'] == 1)),
            ('obj2_rew', (all_obj_df['goal'] == 1) & (all_obj_df['object'] == 2)),
            ('obj2_no_rew', (all_obj_df['goal'] == 0) & (all_obj_df['object'] == 2))
        ]
        tscale = None

        # Process probes: Calculate PETHs for all significant neurons once per condition
        for probe in spikes.keys():
            p_sig_ids = session_neurons.loc[session_neurons['probe'] == probe, 'neuron_id'].values
            if len(p_sig_ids) == 0:
                continue

            probe_peths = {}
            for cond_name, mask in cond_masks:
                res_peths, data = calculate_peths(
                    spikes[probe]['times'], spikes[probe]['clusters'], p_sig_ids,
                    all_obj_df.loc[mask, 'times'], T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING, return_fr=True)
                probe_peths[cond_name] = data
                if tscale is None:
                    tscale = res_peths['tscale']

            # Group into regions
            p_clusters = clusters[probe]
            for region in np.unique(p_clusters['region']):
                if region == 'root': continue
                
                reg_sig_ids = p_sig_ids[np.isin(p_sig_ids, p_clusters['cluster_id'][p_clusters['region'] == region])]
                if len(reg_sig_ids) < MIN_NEURONS:
                    continue
                
                idx = np.where(np.isin(p_sig_ids, reg_sig_ids))[0]
                for cond_name in probe_peths:
                    if region not in obj_data[cond_name]:
                        obj_data[cond_name][region] = probe_peths[cond_name][:, idx, :]
                    else:
                        obj_data[cond_name][region] = np.concatenate([obj_data[cond_name][region], probe_peths[cond_name][:, idx, :]], axis=1)

        # Subtract mean and define baseline
        if tscale is None: continue
        baseline_mask = tscale < -1
        if SUBTRACT_MEAN:
            for cond in obj_data:
                for reg in obj_data[cond]:
                    obj_data[cond][reg] -= np.mean(obj_data[cond][reg], axis=0)

        # Compute pairwise MI between regions
        regions = list(obj_data['obj1_rew'].keys())
        for r1, region_1 in enumerate(regions[:-1]):
            for r2, region_2 in enumerate(regions[r1+1:]):
                print(f'{region_1} - {region_2}')
                
                res_mi = {}
                for cond in ['obj1_rew', 'obj1_no_rew', 'obj2_rew', 'obj2_no_rew']:
                    x_all, y_all = obj_data[cond][region_1], obj_data[cond][region_2]
                    
                    # Parallel compute MI for each time bin
                    results = parallel(delayed(run_mi)(x_all[:, :, tt], y_all[:, :, tt]) for tt in range(len(tscale)))
                    
                    mi_vals = np.array([r[0] for r in results])
                    res_mi[cond] = mi_vals
                    res_mi[f'{cond}_baseline'] = mi_vals - np.nanmean(mi_vals[baseline_mask])

                # Store results
                mi_results.append(pd.DataFrame({
                    **{k: res_mi[k] for k in res_mi},
                    'time': tscale, 'region_1': region_1, 'region_2': region_2,
                    'region_pair': f'{region_1}-{region_2}', 'subject': subject, 'date': date
                }))
                
        # Save progress
        if mi_results:
            pd.concat(mi_results, ignore_index=True).to_csv(
                path_dict['google_drive_data_path'] / f'region_mi_{int(BIN_SIZE*1000)}ms-bins.csv')