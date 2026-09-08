# -*- coding: utf-8 -*-
# %%
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
TEST_BINS = [900, 1200]

# Which neurons to plot
"""
plot_neurons = {
    'region': ['TEa', 'TEa', 'VIS', 'VIS', 'CA1', 'CA1'],
    'subject': ['459601', '459601', '459601', '462910', '462910', '462910'],
    'date': ['20240409', '20240411', '20240411', '20240815', '20240813', '20240814'],
    'probe': ['probe00', 'probe00', 'probe00', 'probe00', 'probe00', 'probe01'],
    'neuron_id': [302, 481, 546, 678, 278, 85]
    }
"""    
plot_neurons = {
    'region': ['TEa', 'TEa', 'VIS', 'CA1'],
    'subject': ['459601', '459601', '459601', '462910'],
    'date': ['20240409', '20240411', '20240411', '20240813'],
    'probe': ['probe00', 'probe00', 'probe00', 'probe00'],
    'neuron_id': [302, 481, 546, 278],
    'y_lim': [6, 3, 8, 2.5]
    }

# Load in data
path_dict = paths()
subjects = load_subjects()    
with open(path_dict['google_drive_data_path'] / 'residuals_position_20mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

# Construct figure
f, axs = plt.subplots(1, len(plot_neurons['region']), figsize=(5.5, 2), dpi=dpi)

# Loop over neurons
for i in range(len(plot_neurons['region'])):

    # Get session info
    this_region = plot_neurons['region'][i]
    this_subject = plot_neurons['subject'][i]
    this_ses = plot_neurons['date'][i]
    this_probe = plot_neurons['probe'][i]
    this_id = plot_neurons['neuron_id'][i]
    y_lim = plot_neurons['y_lim'][i]

    # Get index of this session
    ses_idx = [i for i, (date, probe) in enumerate(zip(spike_dict['date'], spike_dict['probe']))
               if (date == this_ses) & (probe == this_probe)][0]

    # Load in object data
    obj_df = load_objects(this_subject, this_ses)

    # Get which context is the rewarded context for the first and second object
    obj1_goal = obj_df.loc[(obj_df['object'] == 1) & (obj_df['goal'] == 1), 'sound'].values[0]
    obj2_goal = obj_df.loc[(obj_df['object'] == 2) & (obj_df['goal'] == 1), 'sound'].values[0]

    # Get data from this session and region
    spike_counts = spike_dict['residuals'][ses_idx][:, spike_dict['region'][ses_idx] == this_region]  
    neuron_ids = spike_dict['neuron_id'][ses_idx][spike_dict['region'][ses_idx] == this_region]
    spatial_bins = spike_dict['position'][ses_idx]
    context_per_bin = spike_dict['context'][ses_idx]
    unique_bins = np.unique(spatial_bins)
    context_per_trial = context_per_bin[spatial_bins == unique_bins[0]]

    # Reshape spike counts to trials x spatial bins
    this_neuron = spike_counts[:, np.where(neuron_ids == this_id)[0][0]].reshape(
        context_per_trial.shape[0], len(unique_bins))

    plot_df = pd.DataFrame(data={
        'spatial_bin': spatial_bins,
        'context': context_per_bin,
        'residual_fr': spike_counts[:, np.where(neuron_ids == this_id)[0][0]]})

    sns.lineplot(data=plot_df, x='spatial_bin', y='residual_fr', hue='context', ax=axs[i],
                    palette=[colors['context1'], colors['context2']], errorbar='se',
                    err_kws={'lw': 0})
    axs[i].axvspan(TEST_BINS[0], TEST_BINS[1], color='gray', alpha=0.2, lw=0)

    axs[i].set(xticks=[0, 500, 1000, 1500], xticklabels=['0', '50', '100', '150'],
               ylim=[-y_lim, y_lim], yticks=[-y_lim, 0, y_lim], yticklabels=[-y_lim, 0, y_lim],
               ylabel='', xlabel='', title=this_region)
    axs[i].legend().remove()

axs[0].set(ylabel='Residual firing rate (spks/s)')
f.supxlabel('Position (cm)')
sns.despine(trim=True)
plt.subplots_adjust(wspace=0.4, bottom=0.17)
plt.savefig(path_dict['paper_fig_path'] / 'Residuals' / 'example_neurons.pdf')
    
