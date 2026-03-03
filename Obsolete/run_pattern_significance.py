# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:18:07 2026

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from zetapy import zetatstest, zetatstest2
import matplotlib.pyplot as plt
from msvr_functions import paths, load_objects, figure_style, peri_event_trace
colors, dpi = figure_style()

# Settings
SIG_TIME = 2
MIN_RIPPLES = 10
PLOT = True
SMOOTHING_STD = 50  # ms

# Initialize
path_dict = paths()
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
rec = rec.drop_duplicates(['subject', 'date'])
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)

# %% MAIN

pattern_df = pd.DataFrame()
for i, (subject, date) in enumerate(zip(rec['subject'], rec['date'])):
    print(f'Processing {i} of {rec.shape[0]} ({subject} {date})')
    
    # Load in data for this session
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
    trials = pd.read_csv(session_path / 'trials.csv')
    these_ripples = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
    all_obj_df = load_objects(subject, date)
    obj1_goal = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 1), 'times'].values
    obj1_nogoal = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 0), 'times'].values
    obj2_goal = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 1), 'times'].values
    obj2_nogoal = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 0), 'times'].values
    sound1 = trials.loc[trials['soundId'] == 1, 'soundOnsetTime'].values
    sound2 = trials.loc[trials['soundId'] == 2, 'soundOnsetTime'].values
    
    # Get paths to data of this session
    amp_paths = (path_dict['google_drive_data_path'] / 'SpikePatterns').glob(f'{subject}_{date}*.amplitudes.npy')
    
    # Loop over regions and get spike pattern amplitudes    
    amplitudes, times = dict(), dict()
    for amp_path in amp_paths:
        
        # Load in data for this region
        _, _, region = amp_path.stem.split('.')[0].split('_')
        amplitudes[region] = np.load(amp_path)
        times[region] = np.load(amp_path.parent / (amp_path.stem.split('.')[0] + '.times.npy'))
        
    # Loop over regions and get significance
    for j, this_region in enumerate(list(amplitudes.keys())):
        print(f'{this_region}')
        
        # Smooth traces
        activation_rate = amplitudes[this_region]
        time_ax = times[this_region]
        for pat in range(activation_rate.shape[0]):
            activation_rate[pat, :] = gaussian_filter1d(
                activation_rate[pat, :], sigma=SMOOTHING_STD // ((time_ax[1] - time_ax[0]) * 1000))
        
        # Find patterns that encode reward prediction
        p_obj1, p_obj2 = np.empty(activation_rate.shape[0]), np.empty(activation_rate.shape[0])
        z_obj1, z_obj2 = np.empty(activation_rate.shape[0]), np.empty(activation_rate.shape[0])
        for pat in range(activation_rate.shape[0]):
            
            # Run zetatest
            p_obj1[pat], ZETA = zetatstest2(time_ax, activation_rate[pat, :], obj1_goal - SIG_TIME,
                                            time_ax, activation_rate[pat, :], obj1_nogoal - SIG_TIME,
                                            dblUseMaxDur=SIG_TIME)
            z_obj1[pat] = ZETA['dblZETADeviation']
            p_obj2[pat], ZETA = zetatstest2(time_ax, activation_rate[pat, :], obj2_goal - SIG_TIME,
                                            time_ax, activation_rate[pat, :], obj2_nogoal - SIG_TIME,
                                            dblUseMaxDur=SIG_TIME)
            z_obj2[pat] = ZETA['dblZETADeviation']
            
        if PLOT:
            # Plot
            f, axs = plt.subplots(2, activation_rate.shape[0], figsize=(1.75*activation_rate.shape[0], 3.5),
                                  sharex=True, dpi=dpi)
            for pp in range(activation_rate.shape[0]):
                peri_event_trace(activation_rate[pp, :], time_ax,
                                 all_obj_df.loc[all_obj_df['object'] == 1, 'times'],
                                 all_obj_df.loc[all_obj_df['object'] == 1, 'goal'].values + 1,
                                 t_before=2, t_after=1, ax=axs[0, pp],
                                 color_palette=[colors['no-goal'], colors['goal']])
                axs[0, pp].set(xticks=np.arange(-3, 1.5), ylabel='Pattern activation',
                               title=f'p={np.round(p_obj1[pp], 2)}, z={np.round(z_obj1[pp], 1)}')
            for pp in range(activation_rate.shape[0]):
                peri_event_trace(activation_rate[pp, :], time_ax,
                                 all_obj_df.loc[all_obj_df['object'] == 2, 'times'],
                                 all_obj_df.loc[all_obj_df['object'] == 2, 'goal'].values + 1,
                                 t_before=2, t_after=1, ax=axs[1, pp],
                                 color_palette=[colors['no-goal'], colors['goal']])
                axs[1, pp].set(xticks=np.arange(-2, 1.5), ylabel='Pattern activation',
                               xlabel='Time from object entry (s)',
                               title=f'p={np.round(p_obj2[pp], 2)}, z={np.round(z_obj2[pp], 1)}')
            sns.despine(trim=True)
            plt.tight_layout()
                        
            plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns' / f'{this_region}_{subject}_{date}_reward.jpg', dpi=600)
            plt.close(f)
        
        # Find patterns that differentiate between sound1 and sound2 at onset
        p_sound = np.empty(activation_rate.shape[0])
        for pat in range(activation_rate.shape[0]):
            p_sound[pat], _ = zetatstest2(time_ax, activation_rate[pat, :], sound1,
                                          time_ax, activation_rate[pat, :], sound2,
                                          dblUseMaxDur=SIG_TIME)
            
        if PLOT:
            # Plot
            f, axs = plt.subplots(1, activation_rate.shape[0], figsize=(1.75*activation_rate.shape[0], 1.75), dpi=dpi)
            for pp in range(activation_rate.shape[0]):
                peri_event_trace(activation_rate[pp, :], time_ax, trials['soundOnsetTime'], trials['soundId'],
                                 t_before=1, t_after=2, ax=axs[pp],
                                 color_palette=[colors['sound1'], colors['sound2']])
                axs[pp].set(xticks=np.arange(-1, 2.5), ylabel='Pattern activation', title=f'p={np.round(p_sound[pp], 3)}')
            sns.despine(trim=True)
            plt.tight_layout()
            plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns' / f'{this_region}_{subject}_{date}_sound.jpg', dpi=600)
            plt.close(f)
            
        # Find which spike patterns are active during ripples
        these_ripples = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
        p_ripples, z_ripples = np.full(activation_rate.shape[0], np.nan), np.full(activation_rate.shape[0], np.nan)
        if these_ripples.shape[0] >= MIN_RIPPLES:
            
            # Drop last ripple if too close to the end of the recording
            if time_ax[-1] - these_ripples['start_times'].values[-1] < SIG_TIME:
                these_ripples = these_ripples[:-1]   
            
            # Do ZETA
            for pat in range(activation_rate.shape[0]):
                p_ripples[pat], ZETA = zetatstest(time_ax, activation_rate[pat, :],
                                                  these_ripples['start_times'].values - (SIG_TIME / 2),
                                                  dblUseMaxDur=SIG_TIME)
                z_ripples[pat] = ZETA['dblZETADeviation']
            
            if PLOT:
                # Plot
                f, axs = plt.subplots(1, activation_rate.shape[0], figsize=(1.75*activation_rate.shape[0], 1.75), dpi=dpi)
                for pp in range(activation_rate.shape[0]):
                    peri_event_trace(activation_rate[pp, :], time_ax, these_ripples['start_times'],
                                     np.ones(these_ripples.shape[0]),
                                     t_before=1, t_after=1, ax=axs[pp])
                    axs[pp].set(xticks=[-1, 0, 1], ylabel='Pattern activation',
                                title=f'p={np.round(p_ripples[pp], 2)}, z={np.round(z_ripples[pp], 2)}')
                sns.despine(trim=True)
                plt.tight_layout()
                plt.savefig(path_dict['google_drive_fig_path'] / 'SpikePatterns'
                            / f'{this_region}_{subject}_{date}_ripples.jpg', dpi=600)
                plt.close(f)  
        
        # Add to df
        pattern_df = pd.concat((pattern_df, pd.DataFrame(data={
            'p_obj1': p_obj1, 'p_obj2': p_obj2, 'p_sound_id': p_sound, 'p_ripples': p_ripples,
            'z_obj1': z_obj1, 'z_obj2': z_obj2, 'z_ripples': z_ripples,
            'pattern': np.arange(1, activation_rate.shape[0]+1),
            'region': this_region, 'subject': subject, 'date': date
            })))
    
    # Save to disk
    pattern_df.to_csv(path_dict['save_path'] / 'spike_pattern_sig.csv', index=False)
        