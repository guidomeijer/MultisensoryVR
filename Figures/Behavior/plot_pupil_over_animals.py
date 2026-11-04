# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 2026

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from msvr_functions import paths, load_objects, figure_style
colors, dpi = figure_style()

# Settings
T_WINDOW = [0, 2]  # window relative to object entry in which to average pupil size
BASELINE = [-2, -1]  # baseline window relative to object entry
LAST_SES = 5
MIN_TRIALS = 30

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv')
subjects = np.unique(rec['subject']).astype(str)

# Loop over subjects
pupil_obj1, pupil_obj2 = np.full(subjects.shape[0], np.nan), np.full(subjects.shape[0], np.nan)
for i, subject in enumerate(subjects):
    print(f'{subject}')

    # Loop over last sessions
    sessions = list((path_dict['local_data_path'] / 'Subjects' / subject).iterdir())[-LAST_SES:]
    pupil_obj1_ses, pupil_obj2_ses = [], []
    for j, ses_path in enumerate(sessions):

        # Load in data
        if not (ses_path / 'pupil.csv').is_file():
            print(f'No pupil data for {subject} {ses_path.stem}')
            continue
        if not (ses_path / 'trials.csv').is_file():
            continue
        trials = pd.read_csv(ses_path / 'trials.csv')
        if trials.shape[0] < MIN_TRIALS:
            continue
        pupil_df = pd.read_csv(ses_path / 'pupil.csv')
        timestamps = np.load(ses_path / 'camera.times.npy')
        obj_df = load_objects(subject, ses_path.stem)

        # Z-score pupil size per session
        pupil = stats.zscore(pupil_df['width_smooth'].values, nan_policy='omit')

        # Correct for timestamps difference
        if timestamps.shape[0] > pupil.shape[0]:
            timestamps = timestamps[:pupil.shape[0]]
        else:
            pupil = pupil[:timestamps.shape[0]]

        # Loop over trials and get baseline subtracted pupil size
        pupil_dict = {'obj1_goal0': [], 'obj1_goal1': [], 'obj2_goal0': [], 'obj2_goal1': []}
        for k in obj_df[obj_df['object'] != 3].index:
            obj_time = obj_df.loc[k, 'times']
            baseline = np.mean(pupil[(timestamps >= obj_time + BASELINE[0])
                                          & (timestamps < obj_time + BASELINE[1])])
            pupil_dict[f'obj{obj_df.loc[k, "object"]}_goal{obj_df.loc[k, "goal"]}'].append(np.mean(
                pupil[(timestamps >= obj_time + T_WINDOW[0])
                           & (timestamps < obj_time + T_WINDOW[1])]) - baseline)

        # Difference in pupil response between rewarded and unrewarded for this session
        pupil_obj1_ses.append(np.nanmean(pupil_dict['obj1_goal1']) - np.nanmean(pupil_dict['obj1_goal0']))
        pupil_obj2_ses.append(np.nanmean(pupil_dict['obj2_goal1']) - np.nanmean(pupil_dict['obj2_goal0']))

    # Take average over sessions
    if len(pupil_obj1_ses) > 0:
        pupil_obj1[i], pupil_obj2[i] = np.nanmean(pupil_obj1_ses), np.nanmean(pupil_obj2_ses)

# Exclude mice without pupil data
incl = ~np.isnan(pupil_obj1) & ~np.isnan(pupil_obj2)
pupil_obj1, pupil_obj2 = pupil_obj1[incl], pupil_obj2[incl]
print(f'{np.sum(incl)} mice with pupil data')

# %% Stats
_, p_pupil_obj1 = stats.ttest_1samp(pupil_obj1, 0)
_, p_pupil_obj2 = stats.ttest_1samp(pupil_obj2, 0)
_, p_obj1_obj2 = stats.ttest_rel(pupil_obj1, pupil_obj2)
print(f'First object vs 0: p = {p_pupil_obj1:.3f}')
print(f'Second object vs 0: p = {p_pupil_obj2:.3f}')
print(f'First vs second object: p = {p_obj1_obj2:.3f}')

# %% Plot

f, ax1 = plt.subplots(figsize=(1.3, 1.75), dpi=dpi)

ax1.plot([-0.5, 1.5], [0, 0], ls='--', color='grey')
ax1.hlines(np.mean(pupil_obj1), -0.2, 0.2, color='tab:red', lw=1.5)
ax1.hlines(np.mean(pupil_obj2), 0.8, 1.2, color='tab:red', lw=1.5)
ax1.plot([np.zeros_like(pupil_obj1), np.ones_like(pupil_obj2)], [pupil_obj1, pupil_obj2], color='k')
ax1.set(xticks=[0, 1], xticklabels=['First', 'Second'], xlabel='Rewarded object',
        ylabel='Pupil size change (z)', xlim=[-0.5, 1.5])

sns.despine(trim=False)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Behavior' / 'pupil_over_animals.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'Behavior' / 'pupil_over_animals.jpg', dpi=600)
plt.show()
