# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:53:26 2024 by Guido Meijer
"""


import numpy as np
np.random.seed(42)
from os.path import join
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from brainbox.population.decode import get_spike_counts_in_bins, classify
from joblib import Parallel, delayed
from msvr_functions import paths, load_neural_data, load_subjects, load_objects

# Settings
T_BEFORE = 2  # s
T_AFTER = 2
BIN_SIZE = 0.3
STEP_SIZE = 0.025
MIN_NEURONS = 10
N_NEURONS_SUB = 25              # Number of neurons to subselect
N_ITER = 50                     # Number of iterations for subsampling
N_CPUS = 18
MERGE_CORTEX = False

# Create time array
t_centers = np.arange(-T_BEFORE + (BIN_SIZE/2), T_AFTER - ((BIN_SIZE/2) - STEP_SIZE), STEP_SIZE)

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()
kfold_cv = KFold(n_splits=5, shuffle=True, random_state=42)
rec = pd.read_csv(join(path_dict['repo_path'], 'recordings.csv')).astype(str)
clf = RandomForestClassifier(random_state=42, n_jobs=1, n_estimators=20, max_depth=5)

# %% Loop over recordings

def classify_subselection(these_spike_counts, n_sub_neurons, these_trial_labels, this_clf, this_cv):
    """
    Performs classification on a subselection of neurons.
    """
    # Subselect neurons
    these_neurons = np.random.choice(np.arange(these_spike_counts.shape[1]), n_sub_neurons, replace=False)

    # Decode goal vs distractor
    accuracy, _, _ = classify(these_spike_counts[:, these_neurons], these_trial_labels, this_clf, cross_validation=this_cv)
    
    return accuracy

def process_single_session(subject, date, probe, path_dict, kfold_cv, clf, t_centers,
                           BIN_SIZE, MIN_NEURONS, N_NEURONS_SUB, N_ITER, MERGE_CORTEX):
    """
    Processes a single recording session to decode context with subsampled neurons.
    This function is designed to be run in parallel for multiple sessions.
    """
    print(f'Starting session: {subject} {date} {probe}')

    session_decode_df = pd.DataFrame()

    # Load in data
    session_path = path_dict['local_data_path'] / 'Subjects' / f'{subject}' / f'{date}'
    spikes, clusters, channels = load_neural_data(session_path, probe)
    all_obj_df = load_objects(subject, date)

    if MERGE_CORTEX:
        # Change all cortex regions to 'Cortex'
        targets = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC']
        clusters['region'] = np.array(['Cortex' if i in targets else i for i in clusters['region']])

    # Loop over time bins
    for bin_center in t_centers:
        
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
            
            # Get goal coding neurons
            region_neurons = clusters['cluster_id'][clusters['region'] == region]
            if region_neurons.shape[0] < MIN_NEURONS:
                continue

            # Do decoding per object
            accuracy_obj = np.empty(3)
            for ii, obj in enumerate([1, 2, 3]):

                # Select neurons from this region and trials of this object
                region_counts = spike_counts[np.ix_(all_obj_df['object'] == obj, np.isin(neuron_ids, region_neurons))]

                # Get whether this object was a goal or a distractor
                trial_labels = all_obj_df.loc[all_obj_df['object'] == obj, 'sound'].values

                # Check if there are enough neurons for subsampling
                if region_counts.shape[1] < N_NEURONS_SUB:
                    accuracy_obj[ii] = np.nan # Assign NaN if not enough neurons
                    continue

                # Do decoding with random subselection of neurons, now sequentially within this parallel job.
                # The outer loop is parallelized over sessions, so this inner loop runs sequentially
                # within each session's dedicated process.
                iteration_accuracies = []
                for _ in range(N_ITER):
                    acc = classify_subselection(region_counts, N_NEURONS_SUB, trial_labels, clf, kfold_cv)
                    iteration_accuracies.append(acc)
                accuracy_obj[ii] = np.mean(iteration_accuracies)

            # Add to dataframe
            session_decode_df = pd.concat((session_decode_df, pd.DataFrame(data={
                'time': bin_center, 'accuracy': accuracy_obj, 'object': [1, 2, 3],
                'region': region, 'subject': subject, 'date': date, 'probe': probe})))

    return session_decode_df

# %% Main execution block
if __name__ == '__main__':
    # Parallelize over recordings (sessions)
    # random_state ensures reproducible random sequences across parallel jobs
    all_session_results = Parallel(n_jobs=N_CPUS, random_state=42)(
        delayed(process_single_session)(
            subject, date, probe, path_dict, kfold_cv, clf, t_centers,
            BIN_SIZE, MIN_NEURONS, N_NEURONS_SUB, N_ITER, MERGE_CORTEX
        )
        for subject, date, probe in zip(rec['subject'], rec['date'], rec['probe'])
    )

    # Concatenate all results from parallel sessions
    decode_df = pd.concat(all_session_results, ignore_index=True)

    # Save to disk
    if MERGE_CORTEX:
        decode_df.to_csv(path_dict['save_path'] / 'decode_context_subsampled_cortex_time.csv', index=False)
    else:
        decode_df.to_csv(path_dict['save_path'] / 'decode_context_subsampled_time.csv', index=False)
    print('Done')