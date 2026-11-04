# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:57:35 2023

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
from scipy import stats
from scipy.signal import butter, filtfilt
from msvr_functions import (load_subjects, paths, figure_style, event_aligned_averages,
                            load_objects, add_significance)
mne.set_log_level('WARNING')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y

def goal_cluster_test(obj_df):
    goal_matrix = obj_df[obj_df['goal'] == 1].pivot(
        index=['subject', 'date', 'event'], columns='time', values='value').dropna()
    no_goal_matrix = obj_df[obj_df['goal'] == 0].pivot(
        index=['subject', 'date', 'event'], columns='time', values='value').dropna()

    # With two groups the F-statistic equals t^2, so use the squared t-threshold
    df_error = goal_matrix.shape[0] + no_goal_matrix.shape[0] - 2
    f_threshold = stats.t.ppf(1 - 0.2 / 2, df_error) ** 2
    t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
        [goal_matrix.values, no_goal_matrix.values],
        threshold=f_threshold,
        n_permutations=1000,
        tail=1,
        out_type='mask'
    )

    p_values = np.ones(goal_matrix.shape[1])
    for cluster_mask, p_val in zip(clusters, cluster_p_values):
        if p_val < 0.05:
            p_values[cluster_mask] = p_val
    return goal_matrix.columns.values, p_values

def plot_goal_significance(obj_df, ax):
    times, p_values = goal_cluster_test(obj_df)

    # Freeze the y-limits so the significance bar doesn't rescale the axis
    ax.set_ylim(ax.get_ylim())
    add_significance(times, p_values, ax=ax)

def mouse_cluster_test(mouse_df):
    goal_matrix = mouse_df[mouse_df['goal'] == 1].pivot(
        index='subject', columns='time', values='value')
    no_goal_matrix = mouse_df[mouse_df['goal'] == 0].pivot(
        index='subject', columns='time', values='value')

    # Paired test over mice on the rewarded minus unrewarded difference
    goal_matrix, no_goal_matrix = goal_matrix.align(no_goal_matrix, join='inner', axis=0)
    X = goal_matrix.values - no_goal_matrix.values

    t_threshold = stats.t.ppf(1 - 0.2 / 2, X.shape[0] - 1)
    t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        X,
        threshold=t_threshold,
        n_permutations=1000,
        tail=0,
        out_type='mask'
    )

    p_values = np.ones(X.shape[1])
    for cluster_mask, p_val in zip(clusters, cluster_p_values):
        if p_val < 0.05:
            p_values[cluster_mask] = p_val
    return goal_matrix.columns.values, p_values

# Settings
T_BEFORE = 2
T_AFTER = 3
LAST_SES = 5
BIN_SIZE = 0.2
BASELINE = [-2, -1]

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

# Get time
bin_edges = np.round(np.arange(-T_BEFORE, T_AFTER + (BIN_SIZE/2), step=BIN_SIZE), 1)
time_ax = bin_edges[:-1] + (BIN_SIZE/2)

# Load in recording sessions
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)

obj1_df, obj2_df = pd.DataFrame(), pd.DataFrame()
for i, subject in enumerate(np.unique(rec['subject'])):
    print(f'Processing subject {subject}..')
    
    # Loop over last sessions
    sessions = list((path_dict['local_data_path'] / 'Subjects' / subject).iterdir())[-LAST_SES:]

    for j, ses_path in enumerate(sessions):
           
        if not (ses_path / 'pupil.csv').is_file():
            print(f'No pupil data for {subject} {ses_path.stem}')
            continue
        
        # Load in data
        if not (ses_path / 'trials.csv').is_file():
            continue
        trials_df = pd.read_csv(ses_path / 'trials.csv')
        pupil_df = pd.read_csv(ses_path / 'pupil.csv')
        timestamps = np.load(ses_path / 'camera.times.npy')
        all_obj_df = load_objects(subject, ses_path.stem)
    
        # Z-score pupil size per session
        pupil = stats.zscore(pupil_df['width_smooth'].values, nan_policy='omit')

        # Correct for timestamps difference
        if timestamps.shape[0] > pupil.shape[0]:
            timestamps = timestamps[:pupil.shape[0]]
        else:
            pupil = pupil[:timestamps.shape[0]]

        # Do it per object
        obj1_goal_df = event_aligned_averages(
            pupil, timestamps,
            all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['object'] == 1), 'times'].values,
            bin_edges, return_df=True, baseline=BASELINE)
        obj1_no_goal_df = event_aligned_averages(
            pupil, timestamps,
            all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] == 1), 'times'].values,
            bin_edges, return_df=True, baseline=BASELINE)
        obj1_goal_df['goal'] = 1
        obj1_goal_df['subject'] = subject
        obj1_goal_df['date'] = ses_path.stem
        obj1_no_goal_df['goal'] = 0
        obj1_no_goal_df['subject'] = subject
        obj1_no_goal_df['date'] = ses_path.stem
        obj1_df = pd.concat([obj1_df, obj1_goal_df, obj1_no_goal_df], ignore_index=True)
        
        obj2_goal_df = event_aligned_averages(
            pupil, timestamps,
            all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['object'] == 2), 'times'].values,
            bin_edges, return_df=True, baseline=BASELINE)
        obj2_no_goal_df = event_aligned_averages(
            pupil, timestamps,
            all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] == 2), 'times'].values,
            bin_edges, return_df=True, baseline=BASELINE)
        obj2_goal_df['goal'] = 1
        obj2_goal_df['subject'] = subject
        obj2_goal_df['date'] = ses_path.stem
        obj2_no_goal_df['goal'] = 0
        obj2_no_goal_df['subject'] = subject
        obj2_no_goal_df['date'] = ses_path.stem
        obj2_df = pd.concat([obj2_df, obj2_goal_df, obj2_no_goal_df], ignore_index=True)
    
# %%
n_subjects = np.unique(obj1_df['subject']).shape[0]
f, axs = plt.subplots(1, n_subjects, figsize=(1.75*n_subjects, 2), dpi=dpi)
for (i, subject) in enumerate(np.unique(obj1_df['subject'])):
    sns.lineplot(data=obj1_df[obj1_df['subject'] == subject], x='time', y='value', hue='goal',
                 ax=axs[i], hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']],
                 errorbar='se', err_kws={'lw': 0}, legend=None)
    axs[i].set(ylabel='Pupil size (z)', xlabel='Time from object entry (s)',
               title=subject, xticks=[-2, -1, 0, 1, 2, 3])
    plot_goal_significance(obj1_df[obj1_df['subject'] == subject], axs[i])

plt.suptitle('First object')
sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(path_dict['paper_fig_path'] / 'Behavior' /'pupil_per_mouse_obj1.pdf')

# %%

f, axs = plt.subplots(1, n_subjects, figsize=(1.75*n_subjects, 2), dpi=dpi)
for (i, subject) in enumerate(np.unique(obj2_df['subject'])):
    sns.lineplot(data=obj2_df[obj2_df['subject'] == subject], x='time', y='value', hue='goal',
                 ax=axs[i], hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']],
                 errorbar='se', err_kws={'lw': 0}, legend=None)
    axs[i].set(ylabel='Pupil size (z)', xlabel='Time from object entry (s)',
               title=subject, xticks=[-2, -1, 0, 1, 2, 3])
    plot_goal_significance(obj2_df[obj2_df['subject'] == subject], axs[i])

plt.suptitle('Second object')
sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(path_dict['paper_fig_path'] / 'Behavior' / 'pupil_per_mouse_obj2.pdf')

# %% Average within mice and plot over mice

obj1_mouse_df = obj1_df.groupby(['subject', 'goal', 'time'])['value'].mean().reset_index()
obj2_mouse_df = obj2_df.groupby(['subject', 'goal', 'time'])['value'].mean().reset_index()
obj1_times, obj1_p = mouse_cluster_test(obj1_mouse_df)
obj2_times, obj2_p = mouse_cluster_test(obj2_mouse_df)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.5, 1.75), dpi=dpi)

sns.lineplot(data=obj1_mouse_df, x='time', y='value', hue='goal', errorbar='se', err_kws={'lw': 0}, ax=ax1,
             hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']], legend=None)
ax1.set(title='First rewarded object', ylabel='Pupil size (z)', xlabel='', xticks=[-2, -1, 0, 1, 2, 3])
ax1.set_ylim(ax1.get_ylim())
add_significance(obj1_times, obj1_p, ax=ax1)

sns.lineplot(data=obj2_mouse_df, x='time', y='value', hue='goal', errorbar='se', err_kws={'lw': 0}, ax=ax2,
             hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']], legend=None)
ax2.set(title='Second rewarded object', ylabel='', xlabel='', xticks=[-2, -1, 0, 1, 2, 3])
ax2.set_ylim(ax2.get_ylim())
add_significance(obj2_times, obj2_p, ax=ax2)
f.text(0.5, 0.05, 'Time from object entry (s)', ha='center')

sns.despine(trim=True)
plt.subplots_adjust(bottom=0.22, top=0.85, wspace=0.3)

