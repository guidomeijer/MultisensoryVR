# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score
from scipy.stats import sem
from msvr_functions import paths, load_multiple_probes, load_subjects, calculate_peths, load_objects

# Settings
T_BEFORE = 1
T_AFTER = 1
BIN_SIZE = 0.02
SMOOTHING = 0.021

def process_session(subject, date, path_dict):
    """Worker function to process a single session."""
    print(f'Processing: {subject} {date}')
    session_results = []
    
    # Load in object entry times
    all_obj_df = load_objects(subject, date)

    # Load in neural data for all probes
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
    spikes, clusters, channels = load_multiple_probes(session_path, only_good=False, min_fr=0)
    
    # Get list of all regions and which probe they were recorded on
    regions, region_probes = [], []
    for p, probe in enumerate(spikes.keys()):
        regions.append(np.unique(clusters[probe]['region']))
        region_probes.append([probe] * np.unique(clusters[probe]['region']).shape[0])
    
    if len(regions) == 0:
        return pd.DataFrame()
        
    regions = np.concatenate(regions)
    region_probes = np.concatenate(region_probes)
    region_probes = region_probes[regions != 'root']
    regions = regions[regions != 'root']

    for region1, region2 in combinations(regions, 2):
        if region1 == region2:
            continue
        if region1[0] > region2[0]:  # make region pairs alphabetical
            region1, region2 = region2, region1

        for obj_id in [1, 2]:
            # Get MUA of regions
            region1_probe = region_probes[regions == region1][0]
            region1_clusters = clusters[region1_probe]['cluster_id'][clusters[region1_probe]['region'] == region1]
            mua_region1 = spikes[region1_probe]['times'][np.isin(spikes[region1_probe]['clusters'], region1_clusters)]
            
            region2_probe = region_probes[regions == region2][0]
            region2_clusters = clusters[region2_probe]['cluster_id'][clusters[region2_probe]['region'] == region2]
            mua_region2 = spikes[region2_probe]['times'][np.isin(spikes[region2_probe]['clusters'], region2_clusters)]

            # Get binned spikes and subtract mean
            _, binned_spikes1 = calculate_peths(
                mua_region1, np.ones(mua_region1.shape[0]), [1],
                all_obj_df.loc[all_obj_df['object'] == obj_id, 'times'],
                pre_time=T_BEFORE, post_time=T_AFTER, bin_size=BIN_SIZE, smoothing=SMOOTHING)
            
            peth, binned_spikes2 = calculate_peths(
                mua_region2, np.ones(mua_region2.shape[0]), [1],
                all_obj_df.loc[all_obj_df['object'] == obj_id, 'times'],
                pre_time=T_BEFORE, post_time=T_AFTER, bin_size=BIN_SIZE, smoothing=SMOOTHING)
            
            r1_mua, r2_mua = np.squeeze(binned_spikes1), np.squeeze(binned_spikes2)
            r1_centered = r1_mua - np.mean(r1_mua, axis=0)
            r2_centered = r2_mua - np.mean(r2_mua, axis=0)
            
            r1_ss = np.sum(r1_centered ** 2, axis=0)
            r2_ss = np.sum(r2_centered ** 2, axis=0)
            
            epsilon = 1e-8
            corr_matrix = np.dot(r1_centered.T, r2_centered) / (np.sqrt(np.outer(r1_ss, r2_ss)) + epsilon)

            session_results.append({
                'corr_matrix': corr_matrix, 'region1': region1, 'region2': region2, 'object': obj_id,
                'time_ax': peth['tscale'], 'subject': subject, 'date': date
            })
            
    return pd.DataFrame(session_results)

if __name__ == '__main__':
    # Initialize
    path_dict = paths()
    rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
    rec = rec.drop_duplicates(subset=['subject', 'date'])

    # Run sessions in parallel
    print(f'Starting parallel processing of {len(rec)} recordings...')
    results = Parallel(n_jobs=-4)(delayed(process_session)(
        row['subject'], row['date'], path_dict) for _, row in rec.iterrows())

    # Combine results and save
    corr_df = pd.concat(results, ignore_index=True)
    corr_df.to_pickle(path_dict['google_drive_data_path'] / 'corr_mua.pickle')
    print('Done!')