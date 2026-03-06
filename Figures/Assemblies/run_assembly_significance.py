# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 20/02/2026
"""
# %%

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
from zetapy import zetatstest, zetatstest2
import matplotlib.pyplot as plt
from msvr_functions import paths, load_objects, figure_style, peri_event_trace, event_aligned_averages

# Settings
SIG_TIME = 2
MIN_RIPPLES = 10
PLOT = True
N_JOBS = -10

# Initialize
path_dict = paths()
colors, dpi = figure_style()

def process_session(i, subject, date, probe, path_dict, ripples, colors, dpi):
    print(f'Processing {i} ({subject} {date} {probe})')
    assembly_df_session = pd.DataFrame()
    
    try:
        # Load in data for this session
        session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
        trials = pd.read_csv(session_path / 'trials.csv')
        all_obj_df = load_objects(subject, date)
        obj1_goal = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 1), 'times'].values
        obj1_nogoal = all_obj_df.loc[(all_obj_df['object'] == 1) & (all_obj_df['goal'] == 0), 'times'].values
        obj2_goal = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 1), 'times'].values
        obj2_nogoal = all_obj_df.loc[(all_obj_df['object'] == 2) & (all_obj_df['goal'] == 0), 'times'].values
        sound1 = trials.loc[trials['soundId'] == 1, 'soundOnsetTime'].values
        sound2 = trials.loc[trials['soundId'] == 2, 'soundOnsetTime'].values

        # Get paths to data of this session
        amp_paths = list((path_dict['google_drive_data_path'] / 'Assemblies').glob(f'{subject}_{date}_{probe}*.amplitudes.npy'))

        # Loop over regions and get spike pattern amplitudes
        amplitudes, times = dict(), dict()
        for amp_path in amp_paths:
            # Load in data for this region
            region = amp_path.stem.split('.')[0].split('_')[3]
            amplitudes[region] = np.load(amp_path)
            times[region] = np.load(amp_path.parent / (amp_path.stem.split('.')[0] + '.times.npy'))

        # Loop over regions and get significance
        for j, this_region in enumerate(list(amplitudes.keys())):
            
            # Get traces
            activation_rate = amplitudes[this_region]
            time_ax = times[this_region]
            n_assemblies = activation_rate.shape[0]

            # Find patterns that encode reward prediction
            p_obj1, p_obj2 = np.empty(n_assemblies), np.empty(n_assemblies)
            z_obj1, z_obj2 = np.empty(n_assemblies), np.empty(n_assemblies)
            for pat in range(activation_rate.shape[0]):
                # Run zetatest
                p_obj1[pat], ZETA = zetatstest2(time_ax, activation_rate[pat, :], obj1_goal - SIG_TIME,
                                                time_ax, activation_rate[pat, :], obj1_nogoal - SIG_TIME,
                                                dblUseMaxDur=SIG_TIME)
                p_obj2[pat], ZETA = zetatstest2(time_ax, activation_rate[pat, :], obj2_goal - SIG_TIME,
                                                time_ax, activation_rate[pat, :], obj2_nogoal - SIG_TIME,
                                                dblUseMaxDur=SIG_TIME)

            if PLOT & (n_assemblies > 1):
                f, axs = plt.subplots(2, n_assemblies, figsize=(2 * n_assemblies, 5),
                                      sharex=True, dpi=dpi)
                for pp in range(activation_rate.shape[0]):
                    peri_event_trace(activation_rate[pp, :], time_ax,
                                     all_obj_df.loc[all_obj_df['object'] == 1, 'times'],
                                     all_obj_df.loc[all_obj_df['object'] == 1, 'goal'].values + 1,
                                     t_before=2, t_after=1, ax=axs[0, pp],
                                     color_palette=[colors['no-goal'], colors['goal']])
                    axs[0, pp].set(xticks=np.arange(-3, 1.5), ylabel='Assembly activation',
                                   title=f'p={np.round(p_obj1[pp], 2)}, z={np.round(z_obj1[pp], 1)}')
                for pp in range(activation_rate.shape[0]):
                    peri_event_trace(activation_rate[pp, :], time_ax,
                                     all_obj_df.loc[all_obj_df['object'] == 2, 'times'],
                                     all_obj_df.loc[all_obj_df['object'] == 2, 'goal'].values + 1,
                                     t_before=2, t_after=1, ax=axs[1, pp],
                                     color_palette=[colors['no-goal'], colors['goal']])
                    axs[1, pp].set(xticks=np.arange(-2, 1.5), ylabel='Assembly activation',
                                   xlabel='Time from object entry (s)',
                                   title=f'p={np.round(p_obj2[pp], 2)}, z={np.round(z_obj2[pp], 1)}')
                sns.despine(trim=True)
                plt.tight_layout()

                plt.savefig(
                    path_dict['google_drive_fig_path'] / 'Assemblies' / f'{this_region}_{subject}_{date}_{probe}_reward.jpg',
                    dpi=600)
                plt.close(f)

            # Find patterns that differentiate between sound1 and sound2 at onset
            p_sound = np.empty(n_assemblies)
            for pat in range(n_assemblies):
                p_sound[pat], _ = zetatstest2(time_ax, activation_rate[pat, :], sound1,
                                              time_ax, activation_rate[pat, :], sound2,
                                              dblUseMaxDur=SIG_TIME)

            if PLOT & (n_assemblies > 1):
                # Plot
                f, axs = plt.subplots(1, n_assemblies, figsize=(3 * n_assemblies, 3), dpi=dpi)
                for pp in range(n_assemblies):
                    peri_event_trace(activation_rate[pp, :], time_ax, trials['soundOnsetTime'], trials['soundId'],
                                     t_before=1, t_after=2, ax=axs[pp],
                                     color_palette=[colors['sound1'], colors['sound2']])
                    axs[pp].set(xticks=np.arange(-1, 2.5), ylabel='Pattern activation',
                                title=f'p={np.round(p_sound[pp], 3)}')
                sns.despine(trim=True)
                plt.tight_layout()
                plt.savefig(
                    path_dict['google_drive_fig_path'] / 'Assemblies' / f'{this_region}_{subject}_{date}_{probe}_sound.jpg',
                    dpi=600)
                plt.close(f)
            
            # Find which spike patterns are active during ripples
            these_ripples_sess = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
            p_ripples, amp_ripples = np.full(activation_rate.shape[0], np.nan), np.full(activation_rate.shape[0], np.nan)
            if these_ripples_sess.shape[0] >= MIN_RIPPLES:

                # Drop last ripple if too close to the end of the recording
                if time_ax[-1] - these_ripples_sess['start_times'].values[-1] < SIG_TIME:
                    these_ripples_sess = these_ripples_sess[:-1]

                # Loop over assemblies
                for pat in range(activation_rate.shape[0]):

                    # Do ZETA
                    p_ripples[pat], ZETA = zetatstest(time_ax, activation_rate[pat, :],
                                                      these_ripples_sess['start_times'].values - (SIG_TIME / 2),
                                                      dblUseMaxDur=SIG_TIME)

                    # Get mean assembly activation around ripples
                    assembly_ripples = event_aligned_averages(
                        activation_rate[pat, :], time_ax, these_ripples_sess['start_times'].values,
                        timebins=np.arange(-2, 2, time_ax[1] - time_ax[0]), baseline=[-2, -1.5])
                    assembly_ripple_act = np.mean(assembly_ripples, axis=0)

                    # Detect postivive or negative peak
                    is_pos = np.abs(np.max(assembly_ripple_act)) > np.abs(np.min(assembly_ripple_act))
                    if is_pos:
                        amp_ripples[pat] = np.max(assembly_ripple_act)
                    else:
                        amp_ripples[pat] = np.min(assembly_ripple_act)

                if PLOT & (n_assemblies > 1):
                    # Plot
                    f, axs = plt.subplots(1, activation_rate.shape[0], figsize=(3 * activation_rate.shape[0], 3),
                                          dpi=dpi)
                    for pp in range(activation_rate.shape[0]):
                        peri_event_trace(activation_rate[pp, :], time_ax, these_ripples_sess['start_times'],
                                         np.ones(these_ripples_sess.shape[0]),
                                         t_before=1, t_after=1, ax=axs[pp])
                        axs[pp].set(xticks=[-1, 0, 1], xlabel='', ylabel='',
                                    title=f'p={np.round(p_ripples[pp], 2)}, amp={np.round(amp_ripples[pp], 2)}')
                    sns.despine(trim=True)
                    plt.tight_layout()
                    plt.savefig(path_dict['google_drive_fig_path'] / 'Assemblies'
                                / f'{this_region}_{subject}_{date}_{probe}_ripples.jpg', dpi=600)
                    plt.close(f)

            # Add to df
            assembly_df_session = pd.concat((assembly_df_session, pd.DataFrame(data={
                'p_obj1': p_obj1, 'p_obj2': p_obj2, 'p_sound_id': p_sound, 'p_ripples': p_ripples, 'amp_ripples': amp_ripples,
                'assembly': np.arange(1, activation_rate.shape[0] + 1),
                'region': this_region, 'subject': subject, 'date': date, 'probe': probe,
            })))
    except Exception as e:
        print(f"Failed to process {subject} {date} {probe}: {e}")
        
    return assembly_df_session

# %% MAIN

if __name__ == '__main__':
    rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
    ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
    ripples['subject'] = ripples['subject'].astype(str)
    ripples['date'] = ripples['date'].astype(str)

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_session)(i, subject, date, probe, path_dict, ripples, colors, dpi)
        for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe']))
    )
    
    assembly_df = pd.concat(results, ignore_index=True)

    # Save to disk
    assembly_df.to_csv(path_dict['save_path'] / 'assembly_sig.csv', index=False)