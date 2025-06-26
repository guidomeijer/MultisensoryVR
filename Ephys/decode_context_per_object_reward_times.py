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
STEP_SIZE = 0.05
MIN_NEURONS = 5
MIN_TRIALS = 30
ONLY_SIG_NEURONS = False

# Create time array
t_centers = np.arange(-T_BEFORE + (BIN_SIZE/2), T_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))

#clf = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=4, min_samples_leaf=5)
clf = LogisticRegression(solver='liblinear', random_state=42)

# Function for parallelization
def decode_context(bin_center, spikes, use_neurons, all_obj_df):
    
    # Get spike counts per trial for all neurons during this time bin
    these_intervals = np.vstack((all_obj_df['reward_times'] + (bin_center - (BIN_SIZE/2)),
                                 all_obj_df['reward_times'] + (bin_center + (BIN_SIZE/2)))).T
    spike_counts, neuron_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'],
                                                        these_intervals)
    spike_counts = spike_counts.T  # transpose array into [trials x neurons]
        
    # Do decoding per object
    accuracy_obj = np.empty(3)
    for ii, obj in enumerate([1, 2, 3]):
    
        # Select neurons from this region and trials of this object
        use_counts = spike_counts[np.ix_(all_obj_df['object'] == obj, np.isin(neuron_ids, use_neurons))]
        
        # Get whether this object was a goal or a distractor
        trial_labels = all_obj_df.loc[all_obj_df['object'] == obj, 'sound'].values
        
        # Do decoding 
        accuracy_obj[ii], _, _ = classify(use_counts, trial_labels, clf, cross_validation=kfold_cv)
    
    return accuracy_obj


# %% Loop over recordings

decode_df = pd.DataFrame()
for i, (subject, date, probe) in enumerate(zip(rec['subject'], rec['date'], rec['probe'])):
    print(f'\n{subject} {date} {probe} ({i} of {rec.shape[0]})')

    # Load in data
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe)
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    all_obj_df = load_objects(subject, date)
    
    # Generate pseudo reward times for unrewarded objects
    reward_delays = (all_obj_df.loc[all_obj_df['rewarded'] == 1, 'reward_times']
                     - all_obj_df.loc[all_obj_df['rewarded'] == 1, 'times']).values
    all_obj_df.loc[all_obj_df['rewarded'] == 0, 'reward_times'] = (
        all_obj_df.loc[all_obj_df['rewarded'] == 0, 'times']
        + np.random.choice(reward_delays, size=np.sum(all_obj_df['rewarded'] == 0)))
    
    # Get goal coding neurons for this session
    sig_neurons = neurons_df.loc[((neurons_df['sig_context_obj1'] | neurons_df['sig_context_obj2'])
                                  & (neurons_df['subject'] == int(subject))
                                  & (neurons_df['date'] == int(date))
                                  & (neurons_df['probe'] == probe)), 'neuron_id'].values
    
    if trials.shape[0] < MIN_TRIALS:
        continue
    
    # %% Loop over regions
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        print(f'Starting {region}')
        
        # Get region neurons
        region_neurons = clusters['cluster_id'][clusters['region'] == region]
             
        # Use only significant neurons
        if ONLY_SIG_NEURONS:
            use_neurons = region_neurons[np.isin(region_neurons, sig_neurons)]
        else:
            use_neurons = region_neurons
        if use_neurons.shape[0] < MIN_NEURONS:
            continue
        
        # Do decoding with with parallel processing
        results = Parallel(n_jobs=-1)(
            delayed(decode_context)(bin_center, spikes, use_neurons, all_obj_df) for bin_center in t_centers)
        
        # Add to dataframe
        for j in [1, 2, 3]:
            decode_df = pd.concat((decode_df, pd.DataFrame(data={
                'time': t_centers, 'accuracy': [i[j-1] for i in results], 'object': j,
                'region': region, 'subject': subject, 'date': date, 'probe': probe})))
         
    # Save to disk
    decode_df.to_csv(join(path_dict['save_path'], 'decode_context_per_object_reward_times.csv'), index=False)
            
            
          
    