# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:57:35 2023

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
import os
from os.path import join, isfile
import pandas as pd
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style,
                            peri_multiple_events_time_histogram)

# Settings
T_BEFORE = 2
T_AFTER = 4
BIN_SIZE = 0.2
SMOOTHING = 0.1 

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

# Loop over subjects
for i, subject in enumerate(subjects['SubjectID']):

    # List sessions
    sessions = os.listdir(join(data_path, 'Subjects', subject))
    
    # Discard sessions that don't have licks
    no_lick_ses = np.zeros(len(sessions))
    for j, ses in enumerate(sessions):
        if not isfile(join(data_path, 'Subjects', subject, ses, 'lick.times.npy')):
            no_lick_ses[j] = 1
    sessions = np.array(sessions)[~no_lick_ses.astype(int).astype(bool)]
    
    # Create figure for rewards
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4, figsize=(7, 3*np.ceil(len(sessions) / 4)),
                          dpi=dpi, sharey=True, sharex=True)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    for j, ses in enumerate(sessions):

        # Load in data
        trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'))
        lick_times = np.load(join(data_path, 'Subjects', subject, ses, 'lick.times.npy'))

        # Plot
        all_obj_enters = np.concatenate(
            (trials['enterRewardZoneObj1'], trials['enterRewardZoneObj2'], trials['enterRewardZoneObj3']))
        all_obj_ids = np.concatenate(
            (np.ones(trials['enterRewardZoneObj1'].shape),
             np.ones(trials['enterRewardZoneObj2'].shape)*2,
             np.ones(trials['enterRewardZoneObj3'].shape)*3))
        all_obj_ids = all_obj_ids[~np.isnan(all_obj_enters)]
        all_obj_enters = all_obj_enters[~np.isnan(all_obj_enters)] 
        
        peri_multiple_events_time_histogram(
            lick_times, np.ones(lick_times.shape[0]), all_obj_enters, all_obj_ids,
            [1], t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE, smoothing=SMOOTHING, ax=axs[j],
            pethline_kwargs=[{'color': colors['obj1'], 'lw': 1},
                             {'color': colors['obj2'], 'lw': 1},
                             {'color': colors['obj3'], 'lw': 1}],
            errbar_kwargs=[{'color': colors['obj1'], 'alpha': 0.3, 'lw': 0},
                           {'color': colors['obj2'], 'alpha': 0.3, 'lw': 0},
                           {'color': colors['obj3'], 'alpha': 0.3, 'lw': 0}],
            raster_kwargs=[{'color': colors['obj1'], 'lw': 0.5},
                           {'color': colors['obj2'], 'lw': 0.5},
                           {'color': colors['obj3'], 'lw': 0.5}],
            eventline_kwargs={'lw': 0}, include_raster=True)
        
        axs[j].set(ylabel='Licks/s', xticks=np.arange(-2, 7, 2), yticks=[0, np.ceil(axs[j].get_ylim()[1])],
                   title=f'{ses}', xlabel='')
        axs[j].yaxis.set_label_coords(-0.1, 0.75)
        axs[j].plot([0, 0], axs[j].get_ylim(), color='grey', ls='--', lw=0.75, zorder=0)

    f.suptitle(f'{subjects.iloc[i, 1]} ({subject})')
    f.text(0.5, 0.04, 'Time from reward zone entry (s)', ha='center')
    sns.despine(trim=True)
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.8)
    plt.savefig(join(path_dict['fig_path'], f'{subject}_reward_zone_entry_licks.jpg'), dpi=600)