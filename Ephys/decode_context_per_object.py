# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from brainbox.population.decode import get_spike_counts_in_bins, classify
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, load_subjects

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
T_BEFORE = 2  # s
T_AFTER = 2
BIN_SIZE = 0.25
STEP_SIZE = 0.05
N_NEURONS = 30
N_NEURON_PICKS = 100
N_SHUFFLES = 500
ONLY_GOOD_NEURONS = True

# Create time array
t_centers = np.arange(-T_BEFORE + (BIN_SIZE/2), T_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
#clf = RandomForestClassifier(random_state=42)
#clf = GaussianNB()
clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE, histology=True,
                                              only_good=ONLY_GOOD_NEURONS)
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

# %% Functions

def classify_subselection(spike_counts, n_neurons, trial_labels, clf, cv):
    
    # Subselect neurons
    these_neurons = np.random.choice(np.arange(spike_counts.shape[1]), N_NEURONS, replace=False)

    # Decode goal vs distractor
    accuracy, _, _ = classify(spike_counts[:, these_neurons], trial_labels, clf, cross_validation=cv)
    
    return accuracy
    

# %% Loop over time bins
decode_df, shuffles_df = pd.DataFrame(), pd.DataFrame()
for i, bin_center in enumerate(t_centers):
    print(f'Timebin {np.round(bin_center, 2)} ({i} of {len(t_centers)})')
    
    # Get spike counts per trial for all neurons during this time bin
    these_intervals = np.vstack((all_obj_df['times'] + (bin_center - (BIN_SIZE/2)),
                                 all_obj_df['times'] + (bin_center + (BIN_SIZE/2)))).T
    spike_counts, neuron_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'],
                                                        these_intervals)
    spike_counts = spike_counts.T  # transpose array into [trials x neurons]
    
    # Loop over regions
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        if np.sum(clusters['region'] == region) < N_NEURONS:
            continue
        
        # Do decoding per object
        accuracy_obj = np.empty(3)
        for obj in [1, 2, 3]:
        
            # Select neurons from this region and trials of this object
            region_counts = spike_counts[np.ix_(all_obj_df['object'] == obj,
                                                clusters['region'] == region)]
            
            # Get whether this object was a goal or a distractor
            trial_labels = all_obj_df.loc[all_obj_df['object'] == obj, 'sound'].values
            
            # Do decoding with random subselection of neurons, use parallel processing
            results = Parallel(n_jobs=-1)(
                delayed(classify_subselection)(region_counts, N_NEURONS, trial_labels, clf, kfold_cv)
                for i in range(N_NEURON_PICKS))
            accuracy_obj = np.array([result for result in results])
            
            # Add to dataframe
            decode_df = pd.concat((decode_df, pd.DataFrame(data={
                'time': bin_center, 'accuracy': accuracy_obj, 'object': obj, 'region': region})))
        
    # Save to disk
    decode_df.to_csv(join(path_dict['save_path'], 'decode_goal_distractor.csv'), index=False)
            
            
            
            
        
    