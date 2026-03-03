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

# Settings
OVERWRITE = True  # whether to overwrite existing runs
N_PC = 8  # number of PCs to use
MIN_NEURONS = 10  # minimum neurons per region
WIN_SIZE = 0.05  # window size in seconds
SMOOTHING = 0.1  # smoothing of psth
PRE_TIME = 1  # time before stim onset in s
POST_TIME = 1  # time after stim onset in s
SUBTRACT_MEAN = True  # whether to subtract the mean PSTH from each trial (Semedo method)
DIV_BASELINE = False  # whether to divide over baseline + 1 spk/s (Steinmetz method)
K_FOLD = 5  # k in k-fold
MIN_FR = 0.1  # minimum firing rate over the whole recording
N_CORES = -1
MIN_TRIALS = 60

# Paths
path_dict = paths()

# Initialize some things
pca = PCA(n_components=N_PC)
cca = CCA(n_components=1, max_iter=5000)
n_time_bins = int((PRE_TIME + POST_TIME) / WIN_SIZE)
kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)

# %% Function for paralization
def do_cca(act_reg1, act_reg2, tb_1, n_timebins, kfold_obj, cca_obj):
    """
    Performs CCA for a single time bin in region 1 against all time bins in region 2.

    Parameters
    ----------
    act_reg1 : numpy.ndarray
        PCA-transformed activity for the first region. Shape: (n_trials, n_pcs, n_timebins)
    act_reg2 : numpy.ndarray
        PCA-transformed activity for the second region. Shape: (n_trials, n_pcs, n_timebins)
    tb_1 : int
        Index of the time bin to use for the first region.
    n_timebins : int
        Total number of time bins in the analysis window.
    kfold_obj : sklearn.model_selection.KFold
        The initialized KFold object for cross-validation.
    cca_obj : sklearn.cross_decomposition.CCA
        The initialized CCA object.

    Returns
    -------
    tuple
        A tuple containing (r_cca, p_cca), where each is a 1D numpy array
        of length n_timebins.
    """
    r_cca, p_cca = np.empty(n_timebins), np.empty(n_timebins)
    
    for tb_2 in range(n_timebins):
        x_test_all = np.empty(act_reg1.shape[0])
        y_test_all = np.empty(act_reg1.shape[0])
        
        for train_index, test_index in kfold_obj.split(act_reg1[:, :, 0]):
            
            # --- Select train and test data for this fold ---
            X_train, X_test = act_reg1[train_index, :, tb_1], act_reg1[test_index, :, tb_1]
            Y_train, Y_test = act_reg2[train_index, :, tb_2], act_reg2[test_index, :, tb_2]

            # --- Fit CCA and Correct for Sign Ambiguity ---
            cca_obj.fit(X_train, Y_train)
            
            # Transform the training data to check the sign of the learned components
            x_train_trans, y_train_trans = cca_obj.transform(X_train, Y_train)
            
            # Check the sign of the correlation on the training data itself
            sign_flip = 1
            if pearsonr(x_train_trans.T[0], y_train_trans.T[0])[0] < 0:
                sign_flip = -1  # We need to flip the sign for the test data
    
            # --- Transform the held-out test data ---
            x_test_fold, y_test_fold = cca_obj.transform(X_test, Y_test)
            
            # Apply the sign correction to ensure consistency across folds
            x_test_all[test_index] = x_test_fold.T[0]
            y_test_all[test_index] = y_test_fold.T[0] * sign_flip
            
        # Calculate correlation on the full, sign-corrected test data
        r_cca[tb_2], p_cca[tb_2] = pearsonr(x_test_all, y_test_all)
            
    return r_cca, p_cca

# %%
# Load recordings
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv'), dtype=str)
rec = rec.drop_duplicates(subset=['subject', 'date'])

save_path = join(path_dict['save_path'], f'jPECC_goal_{int(WIN_SIZE*1000)}ms-bins.pickle')
if not OVERWRITE and isfile(save_path):
    cca_df = pd.read_pickle(save_path)
else:
    cca_df = pd.DataFrame(columns=['region_pair', 'subject', 'date'])

for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    
    # Load object entries
    all_obj_df = load_objects(subject, date)
    if np.max(all_obj_df['trial_nr']) + 1 < MIN_TRIALS:
        continue
    print(f'\n{subject} {date}')
    
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
                # Get the shape of your binned spike data
                n_trials, n_neurons, n_timebins = binned_spks_goal.shape
                
                # Reshape the data for PCA by concatenating trials and timebins
                binned_reshaped = binned_spks_goal.transpose(0, 2, 1).reshape(-1, n_neurons)
                
                # Fit and transform the data with a single PCA
                pca_transformed = pca.fit_transform(binned_reshaped)
                
                # Reshape the data back to the original format for the CCA
                pca_goal[region] = pca_transformed.reshape(n_trials, n_timebins, N_PC).transpose(0, 2, 1)                    
                
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
                # Get the shape of your binned spike data
                n_trials, n_neurons, n_timebins = binned_spks_dis.shape
                
                # Reshape the data for PCA by concatenating trials and timebins
                binned_reshaped = binned_spks_dis.transpose(0, 2, 1).reshape(-1, n_neurons)
                
                # Fit and transform the data with a single PCA
                pca_transformed = pca.fit_transform(binned_reshaped)
                
                # Reshape the data back to the original format for the CCA
                pca_dis[region] = pca_transformed.reshape(n_trials, n_timebins, N_PC).transpose(0, 2, 1)       
            
    # Perform CCA per region pair
    print('Starting CCA per region pair')
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
            
            # --- Analysis for 'goal' trials ---
            n_timebins = pca_goal[region_1].shape[2]
            results_goal = Parallel(n_jobs=-1)(
                delayed(do_cca)(pca_goal[region_1], pca_goal[region_2], tt, n_timebins, kfold, cca)
                for tt in range(n_timebins))
            r_goal = np.vstack([res[0] for res in results_goal])
            p_goal = np.vstack([res[1] for res in results_goal])
            
            new_row_goal = pd.DataFrame({
                'subject': subject, 'date': date, 'region_1': region_1,
                'region_2': region_2, 'region_pair': region_pair, 'goal': 1,
                'r': [r_goal], 'p': [p_goal], 'time': [psth_goal['tscale']],
            })
            cca_df = pd.concat((cca_df, new_row_goal), ignore_index=True)
            
            # --- Analysis for 'dis' trials ---
            n_timebins_dis = pca_dis[region_1].shape[2]
            results_dis = Parallel(n_jobs=-1)(
                delayed(do_cca)(pca_dis[region_1], pca_dis[region_2], tt, n_timebins_dis, kfold, cca)
                for tt in range(n_timebins_dis))
            
            r_dis = np.vstack([res[0] for res in results_dis])
            p_dis = np.vstack([res[1] for res in results_dis])
            
            new_row_dis = pd.DataFrame({
                'subject': subject, 'date': date, 'region_1': region_1,
                'region_2': region_2, 'region_pair': region_pair, 'goal': 0,
                'r': [r_dis], 'p': [p_dis], 'time': [psth_dis['tscale']],
            })
            cca_df = pd.concat((cca_df, new_row_dis), ignore_index=True)
    
    print(f"Saving results to {save_path}")
    cca_df.to_pickle(save_path)
