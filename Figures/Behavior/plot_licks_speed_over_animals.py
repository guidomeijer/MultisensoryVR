# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:57:35 2023

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
T_BEFORE = [1, 0]
LAST_SES = 5
MIN_TRIALS = 30

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv')
subjects = np.unique(rec['subject']).astype(str)

# Loop over subjects
speed_obj1, speed_obj2 = np.empty(subjects.shape[0]), np.empty(subjects.shape[0])
licks_obj1, licks_obj2 = np.empty(subjects.shape[0]), np.empty(subjects.shape[0])
for i, subject in enumerate(subjects):
    print(f'{subject}')

    # Loop over last sessions
    sessions = list((path_dict['local_data_path'] / 'Subjects' / subject).iterdir())[-LAST_SES:]
    speed_obj1_ses, speed_obj2_ses, lick_obj1_ses, lick_obj2_ses = [], [], [], []
    for j, ses_path in enumerate(sessions):
    
        # Load in data
        if not (ses_path / 'trials.csv').is_file():
            continue
        trials = pd.read_csv(ses_path / 'trials.csv')
        if trials.shape[0] < MIN_TRIALS:
            continue
        lick_times = np.load(ses_path / 'lick.times.npy')
        wheel_times = np.load(ses_path / 'continuous.times.npy')
        wheel_speed = np.load(ses_path / 'continuous.wheelSpeed.npy')
        obj_df = load_objects(subject, ses_path.stem)
                
        # Loop over trials and get speed and licks
        speed_dict = {'obj1_goal0': [], 'obj1_goal1': [], 'obj2_goal0': [], 'obj2_goal1': []}
        lick_dict = {'obj1_goal0': [], 'obj1_goal1': [], 'obj2_goal0': [], 'obj2_goal1': []}
        for k in obj_df[obj_df['object'] != 3].index:
            
            speed_dict[f'obj{obj_df.loc[k, "object"]}_goal{obj_df.loc[k, "goal"]}'].append(np.mean(
                wheel_speed[(wheel_times > obj_df.loc[k, 'times'] - T_BEFORE[0])
                            & (wheel_times < obj_df.loc[k, 'times'] - T_BEFORE[1])]))
            
            lick_dict[f'obj{obj_df.loc[k, "object"]}_goal{obj_df.loc[k, "goal"]}'].append(np.sum(
                (lick_times > obj_df.loc[k, 'times'] - T_BEFORE[0])
                & (lick_times < obj_df.loc[k, 'times'] - T_BEFORE[1])))
            
        # Calculate percentage for this session
        speed_obj1_ses.append(((np.nanmean(speed_dict['obj1_goal0']) - np.nanmean(speed_dict['obj1_goal1']))
                               / np.nanmean(speed_dict['obj1_goal0'])) * 100)
        speed_obj2_ses.append(((np.nanmean(speed_dict['obj2_goal0']) - np.nanmean(speed_dict['obj2_goal1']))
                               / np.nanmean(speed_dict['obj2_goal0'])) * 100)
        
        lick_obj1_ses.append(np.sum(lick_dict['obj1_goal1']) - np.sum(lick_dict['obj1_goal0']))
        lick_obj2_ses.append(np.sum(lick_dict['obj2_goal1']) - np.sum(lick_dict['obj2_goal0']))
        
    # Take average over sessions
    speed_obj1[i], speed_obj2[i] = np.mean(speed_obj1_ses), np.mean(speed_obj2_ses)
    licks_obj1[i], licks_obj2[i] = np.mean(np.abs(lick_obj1_ses)), np.mean(np.abs(lick_obj2_ses))
          
# Invert mice that speed up
speed_obj1[2:4] = np.abs(speed_obj1[2:4])
speed_obj2[2:4] = np.abs(speed_obj2[2:4])
    
# %% Stats
_, p_speed_obj1 = stats.ttest_1samp(speed_obj1, 0)
_, p_speed_obj2 = stats.ttest_1samp(speed_obj2, 0)
_, p_obj1_obj2 = stats.ttest_rel(speed_obj1, speed_obj2)
    
# %% Plot

df_results = pd.DataFrame(data={
    'speed': np.concatenate([speed_obj1, speed_obj2]),
    'licks': np.concatenate([licks_obj1, licks_obj2]),
    'object': np.concatenate([np.ones(subjects.shape[0]), np.ones(subjects.shape[0]) + 1])})

f, ax1 = plt.subplots(figsize=(1.3, 1.75), dpi=dpi)

sns.barplot(data=df_results, x='object', y='speed', ax=ax1, errorbar=None, color='grey', width=0.6)
#sns.boxplot(data=df_results, x='object', y='speed', ax=ax1, color='grey')
ax1.plot([np.zeros_like(speed_obj1), np.ones_like(speed_obj2)], [speed_obj1, speed_obj2], color='k')
ax1.plot([0, 1], [30, 30], color='k')
ax1.text(0.5, 30, '**', fontsize=12, ha='center', va='center')
#ax1.text(0, 10, '**', fontsize=12, ha='center', va='center')
#ax1.text(1, 26, '**', fontsize=12, ha='center', va='center')
ax1.set(xticks=[0, 1], xticklabels=['First', 'Second'], xlabel='Object',
        ylabel='Anticipatory speed change (%)',
        yticks=np.arange(0, 31, 10), ylim=[-5, 30], xlim=[-0.5, 1.5])

sns.despine(trim=False)   
plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'speed_over_animals.pdf')
plt.savefig(path_dict['google_drive_fig_path'] / 'speed_over_animals.jpg', dpi=600)


# %%
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.swarmplot(data=df_results, x='object', y='licks', ax=ax1)

sns.despine(trim=True)   
plt.tight_layout()
plt.savefig(path_dict['fig_path'] / 'licks_over_animals.pdf')
   