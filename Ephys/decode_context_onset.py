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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from brainbox.population.decode import get_spike_counts_in_bins, classify
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, load_subjects, load_objects

# Settings
T_BEFORE = 2  # s
T_AFTER = 2
BIN_SIZE = 0.3
STEP_SIZE = 0.025
N_NEURONS = 20
N_NEURON_PICKS = 100

# Create time array
t_centers = np.arange(-T_BEFORE + (BIN_SIZE/2), T_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
#clf = RandomForestClassifier(random_state=42)
#clf = GaussianNB()
#clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
#clf = LinearDiscriminantAnalysis()
clf = SVC(probability=True)

# %% Functions

def classify_subselection(spike_counts, n_neurons, trial_labels, clf, cv):
    
    # Subselect neurons
    these_neurons = np.random.choice(np.arange(spike_counts.shape[1]), N_NEURONS, replace=False)

    # Decode goal vs distractor
    accuracy, _, _ = classify(spike_counts[:, these_neurons], trial_labels, clf, cross_validation=cv)
    
    return accuracy


# %% Loop over recordings

decode_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\nStarting {subject} {date} {probe}..')

    # Load in data
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)
        
    # %% Loop over time bins
    for i, bin_center in enumerate(t_centers):
        if np.mod(i, 10) == 0:
            print(f'Timebin {np.round(bin_center, 2)} ({i} of {len(t_centers)})')
        
        # Get spike counts per trial for all neurons during this time bin
        these_intervals = np.vstack((trials['soundOnset'] + (bin_center - (BIN_SIZE/2)),
                                     trials['soundOnset'] + (bin_center + (BIN_SIZE/2)))).T
        spike_counts, neuron_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'],
                                                            these_intervals)
        spike_counts = spike_counts.T  # transpose array into [trials x neurons]
        
        # Loop over regions
        for r, region in enumerate(np.unique(clusters['region'])):
            if region == 'root':
                continue
            if np.sum(clusters['region'] == region) < N_NEURONS:
                continue
                        
            # Select neurons from this region and trials of this object
            region_counts = spike_counts[:, clusters['region'] == region]
            
            # Do decoding with random subselection of neurons, use parallel processing
            results = Parallel(n_jobs=-1)(
                delayed(classify_subselection)(region_counts, N_NEURONS, trials['soundId'], clf, kfold_cv)
                for i in range(N_NEURON_PICKS))
            accuracy_sound = np.mean(np.array([result for result in results]))
            
            # Add to dataframe
            decode_df = pd.concat((decode_df, pd.DataFrame(index=[decode_df.shape[0]], data={
                'time': bin_center, 'accuracy': accuracy_sound, 
                'region': region, 'subject': subject, 'date': date, 'probe': probe})))
                
    # Save to disk
    decode_df.to_csv(join(path_dict['save_path'], 'decode_context_onset.csv'), index=False)
            
            
            
            
        
    