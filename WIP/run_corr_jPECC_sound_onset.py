#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:20:47 2024 by Guido Meijer
"""

import numpy as np
np.random.seed(42)  # fix random seed for reproducibility
from os.path import join, isfile
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
#from brainbox.singlecell import calculate_peths
from msvr_functions import paths, load_multiple_probes, calculate_peths

# Settings
OVERWRITE = True  # whether to overwrite existing runs
MIN_NEURONS = 10  # minimum neurons per region
WIN_SIZE = 0.05  # window size in seconds
SMOOTHING = 0.025  # smoothing of psth
PRE_TIME = 0  # time before stim onset in s
POST_TIME = 2  # time after stim onset in s
MIN_FR = 0.1  # minimum firing rate over the whole recording

# Paths
path_dict = paths()

# Initialize some things
n_time_bins = int((PRE_TIME + POST_TIME) / WIN_SIZE)

# %% Function for paralization

def do_corr(act_mat, reg_1, reg_2, tb_1):
    
    r_timebins = np.empty(n_timebins)
    for tb_2 in range(n_timebins):
    
        # Do correlation per neuron pair
        corr_mat = np.empty((act_mat[reg_1].shape[1], act_mat[reg_2].shape[1]))
        for n_1 in range(act_mat[reg_1].shape[1]):
            for n_2 in range(act_mat[reg_2].shape[1]):
                corr_mat[n_1, n_2] = pearsonr(act_mat[reg_1][:, n_1, tb_1], act_mat[reg_2][:, n_2, tb_2])[0]
           
        r_timebins[tb_2] = np.nanmean(corr_mat)
        
    return r_timebins


# %%
# Load recordings
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv'), dtype=str)
rec = rec.drop_duplicates(subset=['subject', 'date'])

if ~OVERWRITE & isfile(join(path_dict['save_path'], 'jPECC_sound_{int(WIN_SIZE*1000)}ms-bins.pickle')):
    cca_df = pd.read_pickle(join(path_dict['save_path'], 'jPECC_sound_{int(WIN_SIZE*1000)}ms-bins.pickle'))
else:
    cca_df = pd.DataFrame(columns=['region_pair', 'subject', 'date'])

for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    
    # Load object entries
    trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', subject, date, 'trials.csv'))
    
    # Load in neural data
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_multiple_probes(session_path, min_fr=MIN_FR)
    
    # Create population activity arrays for all regions
    act_mat = dict()
    for probe in spikes.keys():
        for region in np.unique(clusters[probe]['region']):
    
            # Get spikes and clusters
            clusters_in_region = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
            spks_region = spikes[probe]['times'][np.isin(spikes[probe]['clusters'], clusters_in_region)]
            clus_region = spikes[probe]['clusters'][np.isin(spikes[probe]['clusters'], clusters_in_region)]
  
            if (len(np.unique(clus_region)) >= MIN_NEURONS) & (region != 'root'):
                print(f'Loading population activity for {region}')
  
                # Get PSTH and binned spikes for goal activity
                psth, binned_spks = calculate_peths(
                    spks_region, clus_region, np.unique(clus_region), trials['soundOnset'].values, 
                    pre_time=PRE_TIME, post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING,
                    return_fr=False)
                act_mat[region] = binned_spks
  
    # Perform CCA per region pair
    all_cca_df = pd.DataFrame()
    for r1, region_1 in enumerate(act_mat.keys()):
        for r2, region_2 in enumerate(list(act_mat.keys())[r1:]):
            if region_1 == region_2:
                continue
            region_pair = f'{np.sort([region_1, region_2])[0]}-{np.sort([region_1, region_2])[1]}'
    
            # Skip if already processed
            if (cca_df[(cca_df['region_pair'] == region_pair) & (cca_df['subject'] == subject)
                      & (cca_df['date'] == date)].shape[0] > 0) & ~OVERWRITE:
                print(f'Found {region_1}-{region_2} for {subject} {date}')
                continue
    
            print(f'Running pairwise correlations for region pair {region_pair}')
            n_timebins = act_mat[region_1].shape[2]
            
            results = Parallel(n_jobs=-1)(
                delayed(do_corr)(act_mat, region_1, region_2, tt)
                for tt in range(n_timebins))
            r = np.vstack([i for i in results])
            
            # Add to dataframe
            cca_df = pd.concat((cca_df, pd.DataFrame(index=[cca_df.shape[0]], data={
                'subject': subject, 'date': date, 'region_1': region_1,
                'region_2': region_2, 'region_pair': region_pair, 'goal': 1,
                'r': [r], 'time': [psth['tscale']]})))
            
           
    # Save to disk        
    cca_df.to_pickle(join(path_dict['save_path'], f'jPECC_corr_sound_{int(WIN_SIZE*1000)}ms-bins.pickle'))

