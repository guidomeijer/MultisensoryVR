# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from brainbox.population.decode import classify
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, load_subjects, load_objects, get_spike_counts_in_bins

# Settings
D_BEFORE = 300  # mm
D_AFTER = 300
BIN_SIZE = 50
STEP_SIZE = 7
MIN_NEURONS = 10
MIN_TRIALS = 35
MIN_SPEED = 20  # mm/s
N_CORES = -2

# Create time array
d_centers = np.arange(-D_BEFORE + (BIN_SIZE/2), D_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
kfold_cv = KFold(n_splits=10, shuffle=False)
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
#clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=500))
clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=1))

# Function for parallelization
def decode_context(bin_center, spikes, region_neurons, all_obj_df):

    # Get spike counts per trial for all neurons during this time bin
    these_intervals = np.vstack((all_obj_df['distances'] + (bin_center - (BIN_SIZE/2)),
                                 all_obj_df['distances'] + (bin_center + (BIN_SIZE/2)))).T
    spike_counts, neuron_ids = get_spike_counts_in_bins(spikes['distances'], spikes['clusters'],
                                                        these_intervals)
    spike_counts = spike_counts.T  # transpose array into [trials x neurons]

    # Do decoding per object
    accuracy_obj = np.empty(3)
    for ii, obj in enumerate([1, 2, 3]):

        # Select neurons from this region and trials of this object
        use_counts = spike_counts[np.ix_(all_obj_df['object'] == obj, np.isin(neuron_ids, region_neurons))]

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
    trials = pd.read_csv(join(path_dict['local_data_path'], 'subjects', subject, date, 'trials.csv'))
    if trials.shape[0] < MIN_TRIALS:
        continue
    session_path = join(path_dict['local_data_path'], 'subjects', f'{subject}', f'{date}')
    spikes, clusters, channels = load_neural_data(session_path, probe)
    all_obj_df = load_objects(subject, date)

    # Speed threshold on neural activity
    spikes['distances'] = spikes['distances'][spikes['speeds'] > MIN_SPEED]
    spikes['clusters'] = spikes['clusters'][spikes['speeds'] > MIN_SPEED]

    # %% Loop over regions
    for r, region in enumerate(np.unique(clusters['region'])):
        if region == 'root':
            continue
        print(f'Starting {region}')

        # Get region neurons
        region_neurons = clusters['cluster_id'][clusters['region'] == region]
        if region_neurons.shape[0] < MIN_NEURONS:
            continue

        # Do decoding with with parallel processing
        results = Parallel(n_jobs=N_CORES)(
            delayed(decode_context)(bin_center, spikes, region_neurons, all_obj_df) for bin_center in d_centers)

        # Add to dataframe
        for j in [1, 2, 3]:
            decode_df = pd.concat((decode_df, pd.DataFrame(data={
                'distance': d_centers, 'accuracy': [i[j-1] for i in results], 'object': j,
                'region': region, 'subject': subject, 'date': date, 'probe': probe})))

    # Save to disk
    decode_df.to_csv(join(path_dict['save_path'], 'decode_context_per_object_distance.csv'), index=False)



