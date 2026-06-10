# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 20/02/2026
"""
# %%

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
import matplotlib

matplotlib.use('Agg')
from zetapy import zetatstest, zetatstest2
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from msvr_functions import paths, load_objects, figure_style, peri_event_trace, event_aligned_trace, bin_signal

# Settings
SIG_TIME = 2
MIN_RIPPLES = 10
PLOT = True
N_JOBS = -4
SMOOTHING = 1
RIPPLE_WIN = [-1, 1]
BIN_SIZE = 0.05
OBJ_WIN_CENTER = {'obj1': 0.75, 'obj2': 0}
OBJ_WIN_SIZE = 0.3

# Initialize
path_dict = paths()
colors, dpi = figure_style()


def process_session(i, subject, date, probe, path_dict, ripples, colors, dpi, ripple_win, bin_size):
    print(f'Processing {subject} {date} {probe}')
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
        rewarded_entries = {'obj1': obj1_goal, 'obj2': obj2_goal}
        unrewarded_entries = {'obj1': obj1_nogoal, 'obj2': obj2_nogoal}
        sound1 = trials.loc[trials['soundId'] == 1, 'soundOnsetTime'].values
        sound2 = trials.loc[trials['soundId'] == 2, 'soundOnsetTime'].values

        # Get paths to data of this session
        amp_paths = list(
            (path_dict['google_drive_data_path'] / 'Assemblies').glob(f'{subject}_{date}_{probe}*.amplitudes.npy'))

        # Loop over regions and get spike pattern amplitudes
        amplitudes, times = dict(), dict()
        for amp_path in amp_paths:
            # Load in data for this region
            region = amp_path.stem.split('.')[0].split('_')[3]
            amplitudes[region] = np.load(amp_path)
            times[region] = np.load(amp_path.parent / (amp_path.stem.split('.')[0] + '.times.npy'))

            # Apply Gaussian smoothing to the assembly activation rates
            amplitudes[region] = gaussian_filter1d(amplitudes[region], sigma=SMOOTHING, axis=1)

        # Loop over regions and get significance
        for j, this_region in enumerate(list(amplitudes.keys())):

            # Get traces
            activation_rate = amplitudes[this_region]
            time_ax = times[this_region]
            n_assemblies = activation_rate.shape[0]

            # Smooth traces
            if SMOOTHING > 0:
                activation_rate = gaussian_filter1d(activation_rate, SMOOTHING, axis=1)

            # Find patterns that encode reward prediction
            p_expectation = {'obj1': np.empty(n_assemblies), 'obj2': np.empty(n_assemblies)}
            p_reward = {'obj1': np.empty(n_assemblies), 'obj2': np.empty(n_assemblies)}
            diff_obj = {'obj1': np.empty(n_assemblies), 'obj2': np.empty(n_assemblies)}
            d_prime_obj = {'obj1': np.empty(n_assemblies), 'obj2': np.empty(n_assemblies)}
            for obj in [1, 2]:
                for asm in range(n_assemblies):

                    # Run zetatests
                    p_expectation[f'obj{obj}'][asm], _ = zetatstest2(
                        time_ax, activation_rate[asm, :], rewarded_entries[f'obj{obj}'] - SIG_TIME,
                        time_ax, activation_rate[asm, :], unrewarded_entries[f'obj{obj}'] - SIG_TIME,
                        max_duration=SIG_TIME)
                    p_reward[f'obj{obj}'][asm], _ = zetatstest2(
                        time_ax, activation_rate[asm, :], rewarded_entries[f'obj{obj}'],
                        time_ax, activation_rate[asm, :], unrewarded_entries[f'obj{obj}'],
                        max_duration=SIG_TIME)

                    # Get rewarded-unrewarded difference
                    rewarded_act = bin_signal(
                        time_ax, activation_rate[asm, :], rewarded_entries[f'obj{obj}'] + OBJ_WIN_CENTER[f'obj{obj}'],
                        bin_size=OBJ_WIN_SIZE)
                    unrewarded_act = bin_signal(
                        time_ax, activation_rate[asm, :], unrewarded_entries[f'obj{obj}'] + OBJ_WIN_CENTER[f'obj{obj}'],
                        bin_size=OBJ_WIN_SIZE)
                    diff_obj[f'obj{obj}'][asm] = np.mean(rewarded_act) - np.mean(unrewarded_act)
                    pooled_std = np.sqrt((np.std(rewarded_act)**2 + np.std(unrewarded_act)**2) / 2)
                    d_prime_obj[f'obj{obj}'][asm] = diff_obj[f'obj{obj}'][asm] / pooled_std

            if PLOT & (n_assemblies > 1):
                f, axs = plt.subplots(2, n_assemblies, figsize=(2 * n_assemblies, 7), sharex=True, dpi=dpi)
                for pp in range(activation_rate.shape[0]):
                    peri_event_trace(activation_rate[pp, :], time_ax,
                                     all_obj_df.loc[all_obj_df['object'] == 1, 'times'],
                                     all_obj_df.loc[all_obj_df['object'] == 1, 'goal'].values + 1,
                                     t_before=2, t_after=1, ax=axs[0, pp],
                                     color_palette=[colors['no-goal'], colors['goal']])
                    axs[0, pp].set(xticks=np.arange(-3, 1.5), ylabel='',
                                   title=f'p={np.round(p_expectation["obj1"][pp], 2)}, d={np.round(d_prime_obj["obj1"][pp], 2)}')
                for pp in range(activation_rate.shape[0]):
                    peri_event_trace(activation_rate[pp, :], time_ax,
                                     all_obj_df.loc[all_obj_df['object'] == 2, 'times'],
                                     all_obj_df.loc[all_obj_df['object'] == 2, 'goal'].values + 1,
                                     t_before=2, t_after=1, ax=axs[1, pp],
                                     color_palette=[colors['no-goal'], colors['goal']])
                    axs[1, pp].set(xticks=np.arange(-2, 1.5), xlabel='Time from object entry (s)', ylabel='',
                                   title=f'p={np.round(p_expectation["obj2"][pp], 2)}, d={np.round(d_prime_obj["obj2"][pp], 2)}')
                sns.despine(trim=True)
                plt.tight_layout()

                plt.savefig(
                    path_dict[
                        'google_drive_fig_path'] / 'Assemblies' / f'{this_region}_{subject}_{date}_{probe}_reward.jpg',
                    dpi=600)
                plt.close(f)

            # Find patterns that differentiate between sound1 and sound2 at onset
            p_sound = np.empty(n_assemblies)
            for asm in range(n_assemblies):
                p_sound[asm], _ = zetatstest2(time_ax, activation_rate[asm, :], sound1,
                                              time_ax, activation_rate[asm, :], sound2,
                                              max_duration=SIG_TIME)

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
                    path_dict[
                        'google_drive_fig_path'] / 'Assemblies' / f'{this_region}_{subject}_{date}_{probe}_sound.jpg',
                    dpi=600)
                plt.close(f)

            # Find which assemblies are active during ripples
            these_ripples_sess = ripples[(ripples['subject'] == subject) & (ripples['date'] == date)]
            p_ripples, amp_ripples = np.full(activation_rate.shape[0], np.nan), np.full(activation_rate.shape[0], np.nan)
            if these_ripples_sess.shape[0] >= MIN_RIPPLES:

                # Drop last ripple if too close to the end of the recording
                if time_ax[-1] - these_ripples_sess['start_times'].values[-1] < SIG_TIME:
                    these_ripples_sess = these_ripples_sess[:-1]

                for asm in range(n_assemblies):

                    # Do ZETA
                    p_ripples[asm], ZETA = zetatstest(time_ax, activation_rate[asm, :],
                                                      these_ripples_sess['start_times'].values - (SIG_TIME / 2),
                                                      max_duration=SIG_TIME)

                    # Get mean assembly activation traces at ripples
                    ripple_activation = event_aligned_trace(activation_rate[asm, :], time_ax,
                                                            these_ripples_sess['start_times'].values,
                                                            t_before=np.abs(ripple_win[0]) + 0.5, t_after=ripple_win[1],
                                                            baseline=[-1.5, -1], fs=1/bin_size)

                    # Detect postivive or negative peak
                    is_pos = np.abs(np.max(ripple_activation)) > np.abs(np.min(ripple_activation))
                    if is_pos:
                        amp_ripples[asm] = np.max(ripple_activation)
                    else:
                        amp_ripples[asm] = np.min(ripple_activation)

                if PLOT:
                    # Plot
                    f, axs = plt.subplots(1, n_assemblies, figsize=(3 * activation_rate.shape[0], 3),
                                          dpi=dpi)
                    if len(axs) == 1:
                        axs = [axs]
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
                'p_expectation_obj1': p_expectation['obj1'], 'p_expectation_obj2': p_expectation['obj2'],
                'p_reward_obj1': p_reward['obj1'], 'p_reward_obj2': p_reward['obj2'],
                'diff_obj1': diff_obj['obj1'], 'diff_obj2': diff_obj['obj2'],
                'dprime_obj1': d_prime_obj['obj1'], 'dprime_obj2': d_prime_obj['obj2'],
                'p_sound_id': p_sound, 'p_ripples': p_ripples, 'amp_ripples': amp_ripples,
                'assembly': np.arange(1, activation_rate.shape[0] + 1),
                'region': this_region, 'subject': subject, 'date': date, 'probe': probe,
            })))
    except Exception as e:
        print(f"Failed to process {subject} {date} {probe}: {e}")

    return assembly_df_session


# %% MAIN

if __name__ == '__main__':
    rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv', dtype={'subject': str, 'date': str})
    ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv', dtype={'subject': str, 'date': str})

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_session)(i, subject, date, probe, path_dict, ripples, colors, dpi, RIPPLE_WIN, BIN_SIZE)
        for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe']))
    )

    assembly_df = pd.concat(results, ignore_index=True)

    # Save to disk
    assembly_df.to_csv(path_dict['save_path'] / 'assembly_sig.csv', index=False)