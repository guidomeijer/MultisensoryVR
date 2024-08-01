# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:45:16 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from zetapy import zetatest, zetatest2
from msvr_functions import paths, load_neural_data, load_subjects

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
T_BEFORE = 1  # s
T_AFTER = 2
ALPHA = 0.01

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE, histology=True, only_good=True)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))

# The ZETA test starts looking for significant differences from timepoint zero so we need to 
# subtract the pre time from all the onset times
trials['enterObj1'] = trials['enterObj1'] - 1
trials['enterObj2'] = trials['enterObj2'] - 1
trials['enterObj3'] = trials['enterObj3'] - 1

# Get reward contingencies
sound1_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'Sound1Obj'].values[0]
sound2_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'Sound2Obj'].values[0]
control_obj = subjects.loc[subjects['SubjectID'] == SUBJECT, 'ControlObject'].values[0]

obj1_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 1)[0][0] + 1
obj2_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 2)[0][0] + 1
obj3_goal_sound = np.where(np.array([sound1_obj, sound2_obj, control_obj]) == 3)[0][0] + 1

# Prepare trial data
obj1_df = pd.DataFrame(data={'times': trials['enterObj1'], 'object': 1, 'sound': trials['soundId'],
                             'goal': (trials['soundId'] == obj1_goal_sound).astype(int)})
obj2_df = pd.DataFrame(data={'times': trials['enterObj2'], 'object': 2, 'sound': trials['soundId'],
                             'goal': (trials['soundId'] == obj2_goal_sound).astype(int)})
obj3_df = pd.DataFrame(data={'times': trials['enterObj3'], 'object': 3, 'sound': trials['soundId'],
                             'goal': (trials['soundId'] == obj3_goal_sound).astype(int)})
all_obj_df = pd.concat((obj1_df, obj2_df, obj3_df))
all_obj_df = all_obj_df.sort_values(by='times').reset_index(drop=True)

# Loop over neurons
stats_df = pd.DataFrame()
for n, neuron_id in enumerate(clusters['cluster_id']):
    if np.mod(n, 10) == 0:
        print(f'Processing neuron {n} of {clusters["cluster_id"].shape[0]}')
    sig_goal, sig_control, sig_obj_onset, sig_obj_diff = False, False, False, False
    
    # Get spikes of this neuron
    these_spikes = spikes['times'][spikes['clusters'] == neuron_id]
    if these_spikes.shape[0] == 0:
        continue
    
    # Difference between the two sounds for the first rewarded object
    obj_sound_1 = trials.loc[trials['soundId'] == 1,  f'enterObj{sound1_obj}'].values
    obj_sound_2 = trials.loc[trials['soundId'] == 2,  f'enterObj{sound1_obj}'].values
    obj1_sound_p, _ = zetatest2(these_spikes, obj_sound_1, these_spikes, obj_sound_2,
                                dblUseMaxDur=T_BEFORE + T_AFTER)
    
    # Difference between the two sounds for the second rewarded object
    obj_sound_1 = trials.loc[trials['soundId'] == 1,  f'enterObj{sound2_obj}'].values
    obj_sound_2 = trials.loc[trials['soundId'] == 2,  f'enterObj{sound2_obj}'].values
    obj2_sound_p, _ = zetatest2(these_spikes, obj_sound_1, these_spikes, obj_sound_2,
                                dblUseMaxDur=T_BEFORE + T_AFTER)
    
    # Difference between the two sounds for the control object
    obj_sound_1 = trials.loc[trials['soundId'] == 1,  f'enterObj{control_obj}'].values
    obj_sound_2 = trials.loc[trials['soundId'] == 2,  f'enterObj{control_obj}'].values
    control_sound_p, _ = zetatest2(these_spikes, obj_sound_1, these_spikes, obj_sound_2,
                                   dblUseMaxDur=T_BEFORE + T_AFTER)
    
    # Get neurons which significantly respond to any object appearance 
    obj_p, _, _ = zetatest(these_spikes, all_obj_df['times'].values, dblUseMaxDur=T_AFTER)
    
    # Difference between objects for sound 1
    obj12_sound1_p, _ = zetatest2(these_spikes, 
                                  trials.loc[trials['soundId'] == 1, 'enterObj1'], 
                                  these_spikes,
                                  trials.loc[trials['soundId'] == 1, 'enterObj2'], 
                                  dblUseMaxDur=T_BEFORE + T_AFTER)
    obj13_sound1_p, _ = zetatest2(these_spikes, 
                                  trials.loc[trials['soundId'] == 1, 'enterObj1'], 
                                  these_spikes,
                                  trials.loc[trials['soundId'] == 1, 'enterObj3'], 
                                  dblUseMaxDur=T_BEFORE + T_AFTER)
    obj23_sound1_p, _ = zetatest2(these_spikes, 
                                  trials.loc[trials['soundId'] == 1, 'enterObj2'], 
                                  these_spikes,
                                  trials.loc[trials['soundId'] == 1, 'enterObj3'], 
                                  dblUseMaxDur=T_BEFORE + T_AFTER)
    
    # Difference between objects for sound 2
    obj12_sound2_p, _ = zetatest2(these_spikes, 
                                  trials.loc[trials['soundId'] == 2, 'enterObj1'], 
                                  these_spikes,
                                  trials.loc[trials['soundId'] == 2, 'enterObj2'], 
                                  dblUseMaxDur=T_BEFORE + T_AFTER)
    obj13_sound2_p, _ = zetatest2(these_spikes, 
                                  trials.loc[trials['soundId'] == 2, 'enterObj1'], 
                                  these_spikes,
                                  trials.loc[trials['soundId'] == 2, 'enterObj3'], 
                                  dblUseMaxDur=T_BEFORE + T_AFTER)
    obj23_sound2_p, _ = zetatest2(these_spikes, 
                                  trials.loc[trials['soundId'] == 2, 'enterObj2'], 
                                  these_spikes,
                                  trials.loc[trials['soundId'] == 2, 'enterObj3'], 
                                  dblUseMaxDur=T_BEFORE + T_AFTER)
    
    # Get whether neuron is significant 
    if any(np.array([obj1_sound_p, obj2_sound_p]) < ALPHA):
        sig_goal = True
    if control_sound_p < ALPHA:
        sig_control = True
    if obj_p < ALPHA:
        sig_obj_onset = True
    if ((any(np.array([obj12_sound1_p, obj13_sound1_p, obj23_sound1_p])) < ALPHA)
            | (any(np.array([obj12_sound2_p, obj13_sound2_p, obj23_sound2_p])) < ALPHA)):
        sig_obj_diff = True
        
    # Add to dataframe
    stats_df = pd.concat((stats_df, pd.DataFrame(index=[stats_df.shape[0]], data={
        'neuron_id': neuron_id, 'region': clusters['region'][clusters['cluster_id'] == neuron_id],
        'sig_goal': sig_goal, 'sig_obj_onset': sig_obj_onset, 'sig_control': sig_control,
        'sig_obj_diff': sig_obj_diff, 'p_obj1_sound': obj1_sound_p, 'p_obj2_sound': obj2_sound_p,
        'p_control_sound': control_sound_p, 'p_obj_onset': obj_p,
        'p_obj12_sound1': obj12_sound1_p, 'p_obj23_sound1': obj23_sound1_p, 'p_obj13_sound1': obj13_sound1_p,
        'p_obj12_sound2': obj12_sound2_p, 'p_obj23_sound2': obj23_sound2_p, 'p_obj13_sound2': obj13_sound2_p
        })))
    
# Save to disk
stats_df.to_csv(join(path_dict['save_path'], 'significant_neurons.csv'), index=False)
    
       
    
    
    
