# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 20/02/2026
"""
# %%

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from msvr_functions import paths, load_objects, figure_style, bin_signal

# Settings
MIN_RIPPLES = 10
SMOOTHING = 0.5
OBJ_WIN = 0.3
OBJ_WIN_START = [0.8, 0.8]
RIPPLE_WIN_CENTERS = np.arange(-1.5, 1.1, 0.05)
RIPPLE_WIN = 0.15
N_SHUFFLES = 100
N_CPUS = 20
METRIC = 'cosim'   # dotprod or cosim

# Initialize
path_dict = paths()
colors, dpi = figure_style()

# Load in all ripples
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)

# Load in significant assemblies
assembly_sig = pd.read_csv(path_dict['save_path'] / 'assembly_sig.csv')
assembly_sig['subject'] = assembly_sig['subject'].astype(str)
assembly_sig['date'] = assembly_sig['date'].astype(str)

# Get paths to data of this session
amp_paths = list((path_dict['google_drive_data_path'] / 'Assemblies').glob(f'*amplitudes.npy'))

# Functions
def calc_lda_alignment(obj_amp, y_goal, amplitudes, ripples_ses, time_ax, use_metric, do_shuffle=False):
    n_assemblies = amplitudes.shape[0]

    # Fit LDA to object entry
    obj_lda = LinearDiscriminantAnalysis(n_components=1)
    if do_shuffle:
        y_goal = shuffle(y_goal)
    obj_lda.fit(obj_amp, y_goal)

    # Get LDA axis and norm once
    lda_axis = obj_lda.coef_[0]
    lda_norm = np.linalg.norm(lda_axis)

    # Project ripples to LDA axis
    lda_align_abs, lda_align = np.zeros(RIPPLE_WIN_CENTERS.shape[0]), np.zeros(RIPPLE_WIN_CENTERS.shape[0])
    for k, win_center in enumerate(RIPPLE_WIN_CENTERS):

        # Get assembly amplitudes around ripples
        ripple_amp = np.full((ripples_ses.shape[0], n_assemblies), np.nan)
        for asmbl in range(n_assemblies):
            ripple_amp[:, asmbl] = bin_signal(time_ax, amplitudes[asmbl, :], ripples_ses['start_times'] + win_center,
                                              bin_size=RIPPLE_WIN)

        if use_metric == 'dotprod':

            # Vectorized dot product for all ripples simultaneously
            alignment = np.dot(ripple_amp, lda_axis) / lda_norm

        elif use_metric == 'cosim':
            # Calculate the norm of the ripple activity for each trial at this time bin
            ripple_norm = np.linalg.norm(ripple_amp, axis=1)

            # Prevent division by zero if there are bins with absolutely no activity
            ripple_norm[ripple_norm == 0] = 1e-10

            # Calculate Cosine Similarity: (x dot w) / (||x|| * ||w||)
            alignment = np.dot(ripple_amp, lda_axis) / (ripple_norm * lda_norm)


        # Calculate the mean across ripples for this time bin
        lda_align[k] = np.mean(alignment)

    return lda_align

def process_session(amp_path, ripples, use_metric):
    session_lda_df = pd.DataFrame()
    
    # Load in data for this region
    subject = amp_path.stem.split('.')[0].split('_')[0]
    date = amp_path.stem.split('.')[0].split('_')[1]
    region = amp_path.stem.split('.')[0].split('_')[3]
    print(f'Processing {amp_path.stem}')
    amplitudes = np.load(amp_path)
    time_ax = np.load(amp_path.parent / (amp_path.stem.split('.')[0] + '.times.npy'))
    all_obj_df = load_objects(subject, date)

    # Get ripples for this session
    ripples_ses = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
    if ripples_ses.shape[0] < MIN_RIPPLES:
        return None
    ripples_ses = ripples_ses[ripples_ses['start_times'] < time_ax[-1] - RIPPLE_WIN_CENTERS[-1]]

    n_assemblies = amplitudes.shape[0]
    if n_assemblies == 0:
        return None

    # Apply Gaussian smoothing to the assembly activation rates
    if SMOOTHING > 0:
        amplitudes = gaussian_filter1d(amplitudes, sigma=SMOOTHING, axis=1)

    # Zscore assembly activity
    amplitudes = stats.zscore(amplitudes, axis=1)

    # Loop over object 1 and 2
    for obj in [1, 2]:

        # Fit LDA projection to hit-miss axis during object entries
        # Construct X array
        obj_amp = np.full((all_obj_df[all_obj_df['object'] == obj].shape[0], n_assemblies), np.nan)
        for asmbl in range(n_assemblies):

            obj_goal = all_obj_df.loc[(all_obj_df['object'] == obj) & (all_obj_df['goal'] == 1), 'times'].values
            obj_no_goal = all_obj_df.loc[(all_obj_df['object'] == obj) & (all_obj_df['goal'] == 0), 'times'].values
            obj_goal_amp = bin_signal(time_ax, amplitudes[asmbl, :], obj_goal + OBJ_WIN_START[obj-1] + (OBJ_WIN/2),
                                      bin_size=OBJ_WIN)
            obj_no_goal_amp = bin_signal(time_ax, amplitudes[asmbl, :], obj_no_goal + OBJ_WIN_START[obj-1] + (OBJ_WIN/2),
                                         bin_size=OBJ_WIN)
            obj_goal_no_goal = np.concatenate((obj_goal_amp, obj_no_goal_amp))
            obj_amp[:, asmbl] = obj_goal_no_goal

        # Get trial labels
        y_goal = np.concatenate((np.ones(all_obj_df[(all_obj_df['object'] == obj) & (all_obj_df['goal'] == 1)].shape[0]),
                                 np.zeros(all_obj_df[(all_obj_df['object'] == obj) & (all_obj_df['goal'] == 0)].shape[0])))

        # Get LDA alignment
        lda_align = calc_lda_alignment(obj_amp, y_goal, amplitudes, ripples_ses, time_ax, use_metric, do_shuffle=False)

        # Get LDA alignment for shuffled trial labels (Serial loop within the parallel session)
        lda_align_shuf = np.vstack([
            calc_lda_alignment(obj_amp, y_goal, amplitudes, ripples_ses, time_ax, use_metric, do_shuffle=True)
            for _ in range(N_SHUFFLES)])

        # Do baseline subtraction
        lda_align_bl = lda_align - np.mean(lda_align[RIPPLE_WIN_CENTERS < -1])
        lda_align_bl_shuf = np.zeros_like(lda_align_shuf)
        for kk in range(lda_align_shuf.shape[0]):
            lda_align_bl_shuf[kk, :] = lda_align_shuf[kk, :] - np.mean(lda_align_shuf[kk, RIPPLE_WIN_CENTERS < -1])

        # Add to dataframe
        session_lda_df = pd.concat((session_lda_df, pd.DataFrame(data={
            'lda_align': lda_align, 'lda_align_bl': lda_align_bl,
            'lda_align_shuf': np.mean(lda_align_shuf, axis=0), 'lda_align_bl_shuf': np.mean(lda_align_bl_shuf, axis=0),
            'time_ax': RIPPLE_WIN_CENTERS, 'object': obj,
            'n_assemblies': n_assemblies, 'n_ripples': ripples_ses.shape[0],
            'region': region, 'subject': subject, 'date': date})))
            
    return session_lda_df

# Parallel processing over sessions
results = Parallel(n_jobs=N_CPUS)(
    delayed(process_session)(amp_path, ripples, METRIC)
    for i, amp_path in enumerate(amp_paths))

# Concatenate results from all sessions
results = [res for res in results if res is not None]
lda_df = pd.concat(results, ignore_index=True)
lda_df.to_csv(path_dict['google_drive_data_path'] / f'lda_alignment_{METRIC}_{OBJ_WIN_START[0]}_{OBJ_WIN_START[1]}.csv',
              index=False)
