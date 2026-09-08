# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:28:35 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy import stats
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style, load_objects
colors, dpi = figure_style()

# Settings
DO_PLOT = False
TEST_BINS = [900, 1200]

# Load in data
path_dict = paths()
subjects = load_subjects()    
with open(path_dict['google_drive_data_path'] / 'residuals_position_20mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Loop over recordings
neurons_df = pd.DataFrame()
for i in np.arange(len(spike_dict['date'])):
    print(f'Processing recording {i}/{len(spike_dict["date"])}')

    # Get session info
    this_subject = spike_dict['subject'][i]
    this_ses = spike_dict['date'][i]
    this_probe = spike_dict['probe'][i]

    # Only use FAR or NEAR sessions
    is_far = subjects.loc[subjects['SubjectID'] == this_subject, 'Far'].values[0]
    if is_far != 1:
        continue

    # Load in object data
    obj_df = load_objects(this_subject, this_ses)

    # Get which context is the rewarded context for the first and second object
    obj1_goal = obj_df.loc[(obj_df['object'] == 1) & (obj_df['goal'] == 1), 'sound'].values[0]
    obj2_goal = obj_df.loc[(obj_df['object'] == 2) & (obj_df['goal'] == 1), 'sound'].values[0]

    # Loop over regions
    unique_regions = np.unique(spike_dict['region'][i])
    for region in unique_regions:
        if region == 'root':
            continue

        # Get data from this session and region
        spike_counts = spike_dict['residuals'][i][:, spike_dict['region'][i] == region]  # spatial bins x neurons
        neuron_ids = spike_dict['neuron_id'][i][spike_dict['region'][i] == region]
        spatial_bins = spike_dict['position'][i]
        context_per_bin = spike_dict['context'][i]
        unique_bins = np.unique(spatial_bins)
        context_per_trial = context_per_bin[spatial_bins == unique_bins[0]]

        # Loop over neurons
        p_values = []
        for n, this_id in enumerate(neuron_ids):
            
            # Reshape spike counts to trials x spatial bins
            this_neuron = spike_counts[:, n].reshape(context_per_trial.shape[0], len(unique_bins))

            # Get mean over test window
            test_window_mean = np.mean(
                this_neuron[:, (unique_bins >= TEST_BINS[0]) & (unique_bins <= TEST_BINS[1])],
                axis=1)

            # Do t-test on the two contexts
            _, p_value = stats.ttest_ind(
                test_window_mean[context_per_trial == obj1_goal],
                test_window_mean[context_per_trial == obj2_goal])
            p_values.append(p_value)

            if DO_PLOT and p_value < 0.05:
                plot_df = pd.DataFrame(data={
                    'spatial_bin': spatial_bins,
                    'context': context_per_bin,
                    'residual_fr': spike_counts[:, n]})
                f, ax = plt.subplots(figsize=(2, 2), dpi=dpi)
                sns.lineplot(data=plot_df, x='spatial_bin', y='residual_fr', hue='context', ax=ax,
                             palette=[colors['context1'], colors['context2']], errorbar='se',
                             err_kws={'lw': 0})
                ax.axvspan(TEST_BINS[0], TEST_BINS[1], color='gray', alpha=0.2, lw=0)
                y_max = np.max(np.abs(ax.get_ylim()))
                ax.set(xlabel='Position (cm)', ylabel='Residual firing rate (spks/s)',
                       xticks=[0, 500, 1000, 1500], xticklabels=['0', '50', '100', '150'],
                       ylim=[-y_max, y_max], yticks=[-np.round(y_max), 0, np.round(y_max)])
                ax.legend().remove()

                sns.despine(trim=True)
                plt.tight_layout()
                plt.savefig(path_dict['fig_path'] / 'ContextNeurons' / f'{region}_{this_subject}_{this_ses}_{this_id}.jpg',
                            dpi=600)
                plt.close()

        # Add to dataframe
        neurons_df = pd.concat([
            neurons_df,
            pd.DataFrame({
                'subject': this_subject,
                'date': this_ses,
                'probe': this_probe,
                'region': region,
                'p_value': p_values,
                'neuron_id': spike_dict['neuron_id'][i][spike_dict['region'][i] == region],
                'obj1_goal': obj1_goal,
                'obj2_goal': obj2_goal,
            })
        ], ignore_index=True)

# %% Plot percentage of significant neurons
neurons_df['significant'] = neurons_df['p_value'] < 0.05
per_ses_df = neurons_df.groupby(['region', 'date']).sum(numeric_only=True)
per_ses_df['n_neurons'] = neurons_df.groupby(['region', 'date']).size()
per_ses_df['perc_sig'] = (per_ses_df['significant'] / per_ses_df['n_neurons']) * 100
per_ses_df = per_ses_df.reset_index()

f, ax1 = plt.subplots(figsize=(1.4, 2), dpi=dpi)

this_order = per_ses_df[['region', 'perc_sig']].groupby('region').mean().sort_values(
    'perc_sig', ascending=False).index.values
sns.barplot(data=per_ses_df, x='region', y='perc_sig', ax=ax1, hue='region', errorbar='se',
            palette=colors, order=this_order)
ax1.set(ylabel='Significant neurons (%)',  yticks=[0, 5, 10, 15, 20], xlabel='',
        ylim=[0, 20])
ax1.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()

plt.savefig(path_dict['paper_fig_path'] / 'Residuals' / 'perc_context_neurons.pdf')