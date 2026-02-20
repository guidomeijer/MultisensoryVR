# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:18:07 2026

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import CCA
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from msvr_functions import paths, load_objects, figure_style
colors, dpi = figure_style()

# Settings
MIN_RIPPLES = 10
SMOOTHING_STD = 25  # ms
PLOT = True

# Initialize
path_dict = paths()
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)

# Functions
def time_resolved_cca(amplitudes, times, event_times, fs=100, 
                      pre_event=2.0, post_event=2.0, 
                      step_size=0.05, integration_win=0.3):
    """
    integration_win: The width of the sliding window (in seconds) used to 
                     collect samples for each CCA point.
    step_size: How often to calculate a new CCA point.
    """
    cca_df = pd.DataFrame()
    offsets = np.arange(-pre_event, post_event, step_size)
    regions = list(amplitudes.keys())
    
    # Calculate how many samples per integration window
    n_samples_per_win = int(integration_win * fs)
        
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            reg_a, reg_b = regions[i], regions[j]
            cca_curve = []
            
            for offset in offsets:
                # Collect samples from all 30 events for this specific time-lag
                samples_a = []
                samples_b = []
                
                for t_event in event_times:
                    # Target time relative to event
                    t_target = t_event + offset
                    idx = np.searchsorted(times[reg_a], t_target)
                    
                    # Extract a small window of samples to provide enough data for CCA
                    start, end = idx, idx + n_samples_per_win
                    if end < amplitudes[reg_a].shape[1]:
                        samples_a.append(amplitudes[reg_a][:, start:end])
                        samples_b.append(amplitudes[reg_b][:, start:end])
                
                if not samples_a:
                    cca_curve.append(np.nan)
                    continue

                # Stack samples: (N_events * N_samples_per_win, N_patterns)
                X = np.concatenate(samples_a, axis=1).T
                Y = np.concatenate(samples_b, axis=1).T
                
                # Minimum sample check: samples must be > number of patterns
                if X.shape[0] <= max(X.shape[1], Y.shape[1]):
                    cca_curve.append(np.nan)
                    continue

                cca = CCA(n_components=1)
                Xc, Yc = cca.fit_transform(X, Y)
                cca_curve.append(np.corrcoef(Xc[:, 0], Yc[:, 0])[0, 1])
            
            # Do baseline subtraction
            cca_curve = np.array(cca_curve)
            cca_bl = cca_curve - np.mean(cca_curve[(offsets > -1) & (offsets < 0)])
            
            # Add to dataframe
            cca_df = pd.concat((cca_df, pd.DataFrame(data={
                'cca': np.array(cca_curve), 'cca_bl': cca_bl, 'time': offsets,
                'region_A': reg_a, 'region_B': reg_b, 'region_pair': f'{reg_a}-{reg_b}',
                })))
    return cca_df

# %% MAIN

cca_df = pd.DataFrame()
for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'Processing {i} of {rec.shape[0]} ({subject} {date})')
    
    # Load in data for this session
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
    trials = pd.read_csv(session_path / 'trials.csv')
    these_ripples = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
    obj_df = load_objects(subject, date)
        
    # Get paths to data of this session
    amp_paths = (path_dict['google_drive_data_path'] / 'SpikePatterns').glob(f'{subject}_{date}*.amplitudes.npy')
    
    # Loop over regions and get spike pattern amplitudes    
    amplitudes, times = dict(), dict()
    for amp_path in amp_paths:
        
        # Load in data for this region
        _, _, region = amp_path.stem.split('.')[0].split('_')
        amplitudes[region] = np.load(amp_path)
        times[region] = np.load(amp_path.parent / (amp_path.stem.split('.')[0] + '.times.npy'))
           
        # Smooth traces 
        #for pat in range(amplitudes[region].shape[0]):
        #    amplitudes[region][pat, :] = gaussian_filter1d(
        #        amplitudes[region][pat, :], sigma=SMOOTHING_STD // ((times[region][1] - times[region][0]) * 1000))
    
    if len(amplitudes) == 0:
        continue
    
    # Get PETH for cca
    obj1_df = time_resolved_cca(amplitudes, times, obj_df.loc[obj_df['object'] == 1, 'times'].values)
    obj1_df['event'] = 'obj1'
    obj2_df = time_resolved_cca(amplitudes, times, obj_df.loc[obj_df['object'] == 2, 'times'].values)
    obj2_df['event'] = 'obj2'
    if these_ripples.shape[0] >= MIN_RIPPLES:
        these_ripples = these_ripples[these_ripples['start_times'] < (times[list(times.keys())[0]][-1] - 2)]
        ripple_df = time_resolved_cca(amplitudes, times, these_ripples['start_times'].values)
        ripple_df['event'] = 'ripple'
    else:
        ripple_df = pd.DataFrame()
    cca_df = pd.concat((cca_df, obj1_df, obj2_df, ripple_df))
    cca_df['subject'] = subject
    cca_df['date'] = date
        
# %% Plot

f, axs = plt.subplots(3, 5, figsize=(7, 4), dpi=dpi, sharey=False, sharex=True)
axs = axs.flatten()
plot_df = cca_df[cca_df['event'] == 'obj1']
for i, region_pair in enumerate(plot_df['region_pair'].unique()):
    axs[i].plot([-1, 2], [0, 0], lw=0.5, ls='--')
    sns.lineplot(data=plot_df[plot_df['region_pair'] == region_pair], x='time', y='cca_bl',
                 ax=axs[i], errorbar='se', err_kws={'lw': 0})
    axs[i].set(title=region_pair, xlim=[-1, 2])
sns.despine(trim=True)
plt.tight_layout()

