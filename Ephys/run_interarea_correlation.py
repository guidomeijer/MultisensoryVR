# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:19:43 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from msvr_functions import paths, load_neural_data, load_subjects, calculate_peths

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
T_BEFORE = 1
T_AFTER = 2
BIN_SIZE = 0.025
SMOOTHING = 0.05
SUBTRACT_MEAN = False

# Initialize
path_dict = paths()
subjects = load_subjects()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))

# Get reward contingencies
sound1_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'Sound1Obj'].values[0]
sound2_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'Sound2Obj'].values[0]
control_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'ControlObject'].values[0]

obj1_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 1)[0][0] + 1
obj2_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 2)[0][0] + 1
obj3_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 3)[0][0] + 1

# Prepare trial data
rew_obj1_df = pd.DataFrame(data={'times': trials[f'enterObj{sound1_obj}'],
                                 'object': 1, 'sound': trials['soundId'],
                                 'goal': (trials['soundId'] == obj1_goal_sound).astype(int)})
rew_obj2_df = pd.DataFrame(data={'times': trials[f'enterObj{sound2_obj}'],
                                 'object': 2, 'sound': trials['soundId'],
                                 'goal': (trials['soundId'] == obj2_goal_sound).astype(int)})
control_obj_df = pd.DataFrame(data={'times': trials[f'enterObj{control_obj}'],
                                    'object': 3, 'sound': trials['soundId'],
                                    'goal': (trials['soundId'] == obj3_goal_sound).astype(int)})
all_obj_df = pd.concat((rew_obj1_df, rew_obj2_df, control_obj_df))
all_obj_df = all_obj_df.sort_values(by='times').reset_index(drop=True)

# %% Fuction for parallization

def run_correlation(goal_counts, distractor_counts, tt):
    pairwise_corr = []
    for n1 in range(goal_counts[region_1].shape[1]):  # Neurons in region 1
        for n2 in range(goal_counts[region_2].shape[1]):  # Neurons in region 2
            r, _ = pearsonr(goal_counts[region_1][:, n1, tt],
                            goal_counts[region_2][:, n2, tt])
            pairwise_corr.append(r)
    corr_goal = np.nanmean(pairwise_corr)
    
    pairwise_corr = []
    for n1 in range(distractor_counts[region_1].shape[1]):  # Neurons in region 1
        for n2 in range(distractor_counts[region_2].shape[1]):  # Neurons in region 2
            r, _ = pearsonr(distractor_counts[region_1][:, n1, tt],
                            distractor_counts[region_2][:, n2, tt])
            pairwise_corr.append(r)
    corr_dis = np.nanmean(pairwise_corr)

    return corr_goal, corr_dis


# %%
# Get binned spike counts per region
goal_counts, distractor_counts = dict(), dict()
corr_df = pd.DataFrame()
for j, region in enumerate(np.unique(clusters['region'])):
    if region == 'root':
        continue

    # Get spike counts
    peths, goal_counts[region] = calculate_peths(
        spikes['times'], spikes['clusters'],
        clusters['cluster_id'][clusters['region'] == region],
        rew_obj2_df.loc[rew_obj2_df['goal'] == 1, 'times'], T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING,
        return_fr=False)
    
    peths, distractor_counts[region] = calculate_peths(
        spikes['times'], spikes['clusters'],
        clusters['cluster_id'][clusters['region'] == region],
        rew_obj2_df.loc[rew_obj2_df['goal'] == 0, 'times'], T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING,
        return_fr=False)

    # Get time scale
    tscale = peths['tscale']

# Get pairwise neural correlations between all neuron pairs in both regions
these_regions = list(goal_counts.keys())
for r1, region_1 in enumerate(these_regions[:-1]):
    for r2, region_2 in enumerate(these_regions[r1+1:]):
        print(f'{region_1} - {region_2}')
        
        results = Parallel(n_jobs=-1)(
            delayed(run_correlation)(goal_counts, distractor_counts, tt)
            for tt in range(tscale.shape[0]))
        corr_goal = np.array([result[0] for result in results])
        corr_dis = np.array([result[1] for result in results])
        
        # Baseline subtract        
        corr_goal_bl = corr_goal - np.mean(corr_goal[tscale < 0])
        corr_dis_bl = corr_dis - np.mean(corr_dis[tscale < 0])

        # Add to dataframe
        corr_df = pd.concat((corr_df, pd.DataFrame(data={
            'r_goal': corr_goal, 'r_goal_baseline': corr_goal_bl,
            'r_distractor': corr_dis, 'r_distractor_baseline': corr_dis_bl,
            'time': tscale, 'region_1': region_1, 'region_2': region_2,
            'region_pair': f'{region_1}-{region_2}',
            'subject': SUBJECT, 'date': DATE})), ignore_index=True)

# Save to disk
corr_df.to_csv(join(path_dict['save_path'], f'region_corr_{int(BIN_SIZE*1000)}ms-bins.csv'))