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
from scipy.signal.windows import gaussian
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
#from brainbox.singlecell import calculate_peths
from msvr_functions import paths, load_multiple_probes, load_objects, calculate_peths
cca = CCA(n_components=1, max_iter=5000)
pca = PCA(n_components=10)

# Settings
OVERWRITE = True  # whether to overwrite existing runs
N_PC = 10  # number of PCs to use
MIN_NEURONS = 10  # minimum neurons per region
WIN_SIZE = 0.01  # window size in seconds
SMOOTHING = 0.025  # smoothing of psth
PRE_TIME = 1  # time before stim onset in s
POST_TIME = 0  # time after stim onset in s
SUBTRACT_MEAN = False  # whether to subtract the mean PSTH from each trial (Semedo method)
DIV_BASELINE = True  # whether to divide over baseline + 1 spk/s (Steinmetz method)
K_FOLD = 10  # k in k-fold
MIN_FR = 0.1  # minimum firing rate over the whole recording

# Paths
path_dict = paths()

# Initialize some things
n_time_bins = int((PRE_TIME + POST_TIME) / WIN_SIZE)
kfold = KFold(n_splits=K_FOLD, shuffle=False)

# %% Function for paralization

def do_cca(act_mat, reg_1, reg_2, tb_1):

    n_timebins = pca_goal[region_1].shape[2]
    r_cca, p_cca = np.empty(n_timebins), np.empty(n_timebins)
    
    for tb_2 in range(n_timebins):
    
        # Run CCA
        x_test = np.empty(act_mat[reg_1].shape[0])
        y_test = np.empty(act_mat[reg_1].shape[0])
        for train_index, test_index in kfold.split(act_mat[reg_1][:, :, 0]):
            cca.fit(act_mat[reg_1][train_index, :, tb_1],
                    act_mat[reg_2][train_index, :, tb_2])
            x, y = cca.transform(act_mat[reg_1][test_index, :, tb_1],
                                 act_mat[reg_2][test_index, :, tb_2])
            x_test[test_index] = x.T[0]
            y_test[test_index] = y.T[0]
        r_cca[tb_2], p_cca[tb_2] = pearsonr(x_test, y_test)
            
    return r_cca, p_cca


# %%
# Load recordings
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv'), dtype=str)
rec = rec.drop_duplicates(subset=['subject', 'date'])

if ~OVERWRITE & isfile(join(path_dict['save_path'], 'jPECC_goal_{int(WIN_SIZE*1000)}ms-bins.pickle')):
    cca_df = pd.read_pickle(join(path_dict['save_path'], 'jPECC_goal_{int(WIN_SIZE*1000)}ms-bins.pickle'))
else:
    cca_df = pd.DataFrame(columns=['region_pair', 'subject', 'date'])

for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    
    # Load object entries
    all_obj_df = load_objects(subject, date)
    
    # Load in neural data
    session_path = join(path_dict['local_data_path'], 'Subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_multiple_probes(session_path, min_fr=MIN_FR)
    
    # Create population activity arrays for all regions
    pca_goal, pca_dis, spks_goal, spks_dis = dict(), dict(), dict(), dict()
    for probe in spikes.keys():
        for region in np.unique(clusters[probe]['region']):
    
            # Get spikes and clusters
            clusters_in_region = clusters[probe]['cluster_id'][clusters[probe]['region'] == region]
            spks_region = spikes[probe]['times'][np.isin(spikes[probe]['clusters'], clusters_in_region)]
            clus_region = spikes[probe]['clusters'][np.isin(spikes[probe]['clusters'], clusters_in_region)]
  
            if (len(np.unique(clus_region)) >= MIN_NEURONS) & (region != 'root'):
                print(f'Loading population activity for {region}')
  
                # Get PSTH and binned spikes for goal activity
                goal_entries = np.sort(all_obj_df.loc[
                    (all_obj_df['goal'] == 1) & (all_obj_df['object'] == 2), 'times'].values)
                psth_goal, binned_spks_goal = calculate_peths(
                    spks_region, clus_region, np.unique(clus_region), goal_entries, 
                    pre_time=PRE_TIME, post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING,
                    return_fr=False)
  
                if DIV_BASELINE:
                    # Divide each trial over baseline + 1 spks/s
                    for nn in range(binned_spks_goal.shape[1]):
                        for tt in range(binned_spks_goal.shape[0]):
                            binned_spks_goal[tt, nn, :] = (binned_spks_goal[tt, nn, :]
                                                          / (np.median(psth_goal['means'][nn, psth_goal['tscale'] < 0])
                                                             + (1/PRE_TIME)))
  
                if SUBTRACT_MEAN:
                    # Subtract mean PSTH from each opto stim
                    for tt in range(binned_spks_goal.shape[0]):
                        binned_spks_goal[tt, :, :] = binned_spks_goal[tt, :, :] - psth_goal['means']
  
                # Add to dict
                spks_goal[region] = binned_spks_goal
  
                # Perform PCA
                pca_goal[region] = np.empty([binned_spks_goal.shape[0], N_PC, binned_spks_goal.shape[2]])
                for tb in range(binned_spks_goal.shape[2]):
                    pca_goal[region][:, :, tb] = pca.fit_transform(binned_spks_goal[:, :, tb])
                
                # Get PSTH and binned spikes for distractor activity
                dis_entries = np.sort(all_obj_df.loc[
                    (all_obj_df['goal'] == 0) & (all_obj_df['object'] == 2), 'times'].values)
                psth_dis, binned_spks_dis = calculate_peths(
                    spks_region, clus_region, np.unique(clus_region), dis_entries, 
                    pre_time=PRE_TIME, post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING,
                    return_fr=False)
  
                if DIV_BASELINE:
                    # Divide each trial over baseline + 1 spks/s
                    for nn in range(binned_spks_dis.shape[1]):
                        for tt in range(binned_spks_dis.shape[0]):
                            binned_spks_dis[tt, nn, :] = (binned_spks_dis[tt, nn, :]
                                                          / (np.median(psth_dis['means'][nn, psth_dis['tscale'] < 0])
                                                             + (1/PRE_TIME)))
  
                if SUBTRACT_MEAN:
                    # Subtract mean PSTH from each opto stim
                    for tt in range(binned_spks_dis.shape[0]):
                        binned_spks_dis[tt, :, :] = binned_spks_dis[tt, :, :] - psth_dis['means']
  
                # Add to dict
                spks_dis[region] = binned_spks_dis
  
                # Perform PCA
                pca_dis[region] = np.empty([binned_spks_dis.shape[0], N_PC, binned_spks_dis.shape[2]])
                for tb in range(binned_spks_dis.shape[2]):
                    pca_dis[region][:, :, tb] = pca.fit_transform(binned_spks_dis[:, :, tb])
    
    # Perform CCA per region pair
    print('Starting CCA per region pair')
    all_cca_df = pd.DataFrame()
    for r1, region_1 in enumerate(pca_goal.keys()):
        for r2, region_2 in enumerate(list(pca_goal.keys())[r1:]):
            if region_1 == region_2:
                continue
            region_pair = f'{np.sort([region_1, region_2])[0]}-{np.sort([region_1, region_2])[1]}'
    
            # Skip if already processed
            if (cca_df[(cca_df['region_pair'] == region_pair) & (cca_df['subject'] == subject)
                      & (cca_df['date'] == date)].shape[0] > 0) & ~OVERWRITE:
                print(f'Found {region_1}-{region_2} for {subject} {date}')
                continue
    
            print(f'Running CCA for region pair {region_pair}')
            n_timebins = pca_goal[region_1].shape[2]
            
            results = Parallel(n_jobs=-1)(
                delayed(do_cca)(pca_goal, region_1, region_2, tt)
                for tt in range(n_timebins))
            r_goal = np.vstack([i[0] for i in results])
            p_goal = np.vstack([i[1] for i in results])

            # Add to dataframe
            cca_df = pd.concat((cca_df, pd.DataFrame(index=[cca_df.shape[0]], data={
                'subject': subject, 'date': date, 'region_1': region_1,
                'region_2': region_2, 'region_pair': region_pair, 'goal': 1,
                'r': [r_goal], 'p': [p_goal], 'time': [psth_goal['tscale']]})))
            
            results = Parallel(n_jobs=-1)(
                delayed(do_cca)(pca_dis, region_1, region_2, tt)
                for tt in range(n_timebins))
            r_dis = np.vstack([i[0] for i in results])
            p_dis = np.vstack([i[1] for i in results])
            
            # Add to dataframe
            cca_df = pd.concat((cca_df, pd.DataFrame(index=[cca_df.shape[0]], data={
                'subject': subject, 'date': date, 'region_1': region_1,
                'region_2': region_2, 'region_pair': region_pair, 'goal': 0,
                'r': [r_dis], 'p': [p_dis], 'time': [psth_goal['tscale']]})))

    # Save to disk        
    cca_df.to_pickle(join(path_dict['save_path'], f'jPECC_goal_{int(WIN_SIZE*1000)}ms-bins.pickle'))

