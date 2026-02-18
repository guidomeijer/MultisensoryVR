# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:18:07 2026

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from msvr_functions import paths, load_objects, figure_style, load_subjects
colors, dpi = figure_style()

# Settings
SMOOTHING_STD = 100  # ms
N_BINS = 50
PLOT = True

# Initialize
path_dict = paths()
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)
subjects = load_subjects()


# %% MAIN

cca_df = pd.DataFrame()
for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'Processing {i} of {rec.shape[0]} ({subject} {date})')
    
    # Load in data for this session
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
    trials = pd.read_csv(session_path / 'trials.csv')
    these_ripples = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
    obj_df = load_objects(subject, date)
    is_far = subjects.loc[subjects['SubjectID'] == subject, 'Far'].values[0]    
    
    # Get paths to data of this session
    amp_paths = (path_dict['google_drive_data_path'] / 'SpikePatterns').glob(f'{subject}_{date}*.amplitudes.npy')
    
    # Loop over regions and get spike pattern amplitudes    
    for amp_path in amp_paths:
        
        # Load in data for this region
        _, _, region = amp_path.stem.split('.')[0].split('_')
        amplitudes = np.load(amp_path)
        times = np.load(amp_path.parent / (amp_path.stem.split('.')[0] + '.times.npy'))
           
        # Loop over patterns
        time_warped_amp = np.full((trials.shape[0], N_BINS, amplitudes.shape[0]), np.nan)
        for pat in range(amplitudes.shape[0]):
            
            # Smooth traces 
            amplitudes[pat, :] = gaussian_filter1d(
                amplitudes[pat, :], sigma=SMOOTHING_STD // ((times[1] - times[0]) * 1000))
            
            # Loop over trials 
            for t, (env_start, env_end) in enumerate(zip(trials['enterEnvTime'], trials['exitEnvTime'])):
                
                # Get time warped amplitudes
                bin_edges = np.linspace(env_start, env_end, num=N_BINS+1)
                time_warped_amp[t, :, pat], _, _ = stats.binned_statistic(
                    times, amplitudes[pat, :], bins=bin_edges)
        
        # Plot
        if region == 'AUD':
            f, axs = plt.subplots(2, 4, figsize=(7, 4), dpi=dpi)
        else:
            f, axs = plt.subplots(2, 3, figsize=(7, 4), dpi=dpi)
        axs = axs.flatten()
        for pat in range(amplitudes.shape[0]):
            
            context_1 = time_warped_amp[trials['soundId'] == 1, :, pat]
            context_2 = time_warped_amp[trials['soundId'] == 2, :, pat]
            axs[pat].fill_between(
                np.arange(1, N_BINS+1),
                np.mean(context_1, axis=0) - stats.sem(context_1, axis=0),
                np.mean(context_1, axis=0) + stats.sem(context_1, axis=0),
                color=colors['sound1'], alpha=0.25, lw=0
                )
            axs[pat].plot(np.arange(1, N_BINS+1), np.mean(context_1, axis=0), color=colors['sound1'])
            
            axs[pat].fill_between(
                np.arange(1, N_BINS+1),
                np.mean(context_2, axis=0) - stats.sem(context_2, axis=0),
                np.mean(context_2, axis=0) + stats.sem(context_2, axis=0),
                color=colors['sound2'], alpha=0.25, lw=0
                )
            axs[pat].plot(np.arange(1, N_BINS+1), np.mean(context_2, axis=0), color=colors['sound2'])
            
            axs[pat].plot([N_BINS * 0.3, N_BINS * 0.3], axs[pat].get_ylim(), lw=0.5, ls='--', color='grey')
            if is_far:
                axs[pat].plot([N_BINS * 0.9, N_BINS * 0.9], axs[pat].get_ylim(), lw=0.5, ls='--', color='grey')
            else:
                axs[pat].plot([N_BINS * 0.6, N_BINS * 0.6], axs[pat].get_ylim(), lw=0.5, ls='--', color='grey')
            axs[pat].set(xticks=[1, N_BINS])
        
        f.text(0.5, 0.04, 'Warped time', ha='center')
        f.text(0.04, 0.5, 'Spike pattern amplitude', ha='center', va='center', rotation='vertical')
        f.suptitle(region)
        sns.despine(trim=True)
        plt.subplots_adjust(left=0.11, bottom=0.12, right=0.98, top=0.95, wspace=0.15, hspace=0.2)
        
        plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns' / 'Timewarped' / f'{subject}_{date}_{region}.jpg', dpi=600)
        plt.close(f)

                