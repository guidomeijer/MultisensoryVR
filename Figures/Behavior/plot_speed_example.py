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
from msvr_functions import (load_subjects, paths, figure_style, event_aligned_averages,
                            load_objects, add_significance)
mne.set_log_level('WARNING')

# Settings
T_BEFORE = 2
T_AFTER = 3
BIN_SIZE = 0.2

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
subject = '466395'
both_obj_df, obj1_df, obj2_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
# Take slice of dataframe
sub_rec = rec[rec['subject'] == subject]

for j, date in enumerate(np.unique(sub_rec['date'])):

    # Get reward contingencies
    sound1_obj = subjects.loc[subjects['SubjectID'] == subject, "Sound1Obj"].values[0]
    sound2_obj = subjects.loc[subjects['SubjectID'] == subject, "Sound2Obj"].values[0]
    control_obj = subjects.loc[subjects['SubjectID'] == subject, "ControlObject"].values[0]    
   
    # Load in data
    trials = pd.read_csv(data_path / 'Subjects' / subject / date / 'trials.csv')
    wheel_times = np.load(data_path / 'Subjects' / subject / date / 'continuous.times.npy')
    wheel_speed = np.load(data_path / 'Subjects' / subject / date / 'continuous.wheelSpeed.npy')
    all_obj_df = load_objects(subject, date)

    # Convert to cm/s and downsample
    wheel_speed = wheel_speed[::50] / 10
    wheel_times = wheel_times[::50]
    
    # Get average goal and no goal speed for both objects
    this_goal_df = event_aligned_averages(
        wheel_speed, wheel_times,
        all_obj_df.loc[all_obj_df['goal'] == 1, 'times'].values,
        bin_edges, return_df=True)
    this_no_goal_df = event_aligned_averages(
        wheel_speed, wheel_times,
        all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] != 3), 'times'].values,
        bin_edges, return_df=True)
    
    # Add to subject dataframe
    this_goal_df['goal'] = 1
    this_goal_df['subject'] = subject
    this_no_goal_df['goal'] = 0
    this_no_goal_df['subject'] = subject
    both_obj_df = pd.concat([both_obj_df, this_goal_df, this_no_goal_df], ignore_index=True)
    
    # Do it per object
    obj1_goal_df = event_aligned_averages(
        wheel_speed, wheel_times,
        all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['object'] == 1), 'times'].values,
        bin_edges, return_df=True)
    obj1_no_goal_df = event_aligned_averages(
        wheel_speed, wheel_times,
        all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] == 1), 'times'].values,
        bin_edges, return_df=True)
    obj1_goal_df['goal'] = 1
    obj1_goal_df['subject'] = subject
    obj1_goal_df['date'] = date
    obj1_no_goal_df['goal'] = 0
    obj1_no_goal_df['subject'] = subject
    obj1_no_goal_df['date'] = date
    obj1_df = pd.concat([obj1_df, obj1_goal_df, obj1_no_goal_df], ignore_index=True)
    
    obj2_goal_df = event_aligned_averages(
        wheel_speed, wheel_times,
        all_obj_df.loc[(all_obj_df['goal'] == 1) & (all_obj_df['object'] == 2), 'times'].values,
        bin_edges, return_df=True)
    obj2_no_goal_df = event_aligned_averages(
        wheel_speed, wheel_times,
        all_obj_df.loc[(all_obj_df['goal'] == 0) & (all_obj_df['object'] == 2), 'times'].values,
        bin_edges, return_df=True)
    obj2_goal_df['goal'] = 1
    obj2_goal_df['subject'] = subject
    obj2_goal_df['date'] = date
    obj2_no_goal_df['goal'] = 0
    obj2_no_goal_df['subject'] = subject
    obj2_no_goal_df['date'] = date
    obj2_df = pd.concat([obj2_df, obj2_goal_df, obj2_no_goal_df], ignore_index=True)


# %% Cluster-based permutation test rewarded vs unrewarded object entries

def goal_cluster_test(obj_df):
    goal_matrix = obj_df[obj_df['goal'] == 1].pivot(
        index=['date', 'event'], columns='time', values='value').dropna()
    no_goal_matrix = obj_df[obj_df['goal'] == 0].pivot(
        index=['date', 'event'], columns='time', values='value').dropna()

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

obj1_times, obj1_p = goal_cluster_test(obj1_df)
obj2_times, obj2_p = goal_cluster_test(obj2_df)

# %%

f, axs = plt.subplots(1, 2, figsize=(2.5, 1.75), dpi=dpi, sharey=True)

axs[0].plot([0, 0], [4, 21], ls='--', color='grey', lw=0.5)
sns.lineplot(data=obj1_df[obj1_df['subject'] == subject], x='time', y='value', hue='goal',
             ax=axs[0], hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']],
             errorbar='se', err_kws={'lw': 0}, legend=None)
axs[0].set(ylabel='Running speed (cm/s)', xlabel='',
           title='First rewarded object', xticks=[-2, -1, 0, 1, 2, 3], yticks=[5, 10, 15, 20])

axs[1].plot([0, 0], [4, 21], ls='--', color='grey', lw=0.5)
sns.lineplot(data=obj2_df[obj2_df['subject'] == subject], x='time', y='value', hue='goal',
             ax=axs[1], hue_order=[1, 0], palette=[colors['goal'], colors['no-goal']],
             errorbar='se', err_kws={'lw': 0}, legend=None)
axs[1].set(title='Second rewarded object', xticks=[-2, -1, 0, 1, 2, 3], yticks=[5, 10, 15, 20], xlabel='',
           ylim=[4.8, 20])
add_significance(obj1_times, obj1_p, ax=axs[0], y_pos=19)
add_significance(obj2_times, obj2_p, ax=axs[1], y_pos=19)
#handles, previous_labels = axs[1].get_legend_handles_labels()
#axs[1].legend(handles=handles, labels=['Rewarded', 'Unrewarded'], title='',
#              bbox_to_anchor=[0.4, 0.7])


f.text(0.5, 0.05, 'Time from object entry (s)', ha='center')

sns.despine(trim=True)
plt.subplots_adjust(bottom=0.22, top=0.85)

plt.savefig(path_dict['paper_fig_path'] / 'Behavior' / 'example_mouse_running.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'Behavior' / 'example_mouse_running.jpg', dpi=600)