# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:57:35 2023

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from os.path import join, isfile
import pandas as pd
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style,
                            peri_multiple_events_time_histogram)

# Settings
CM_BEFORE = 20
CM_AFTER = 50
BIN_SIZE = 0.2
SMOOTHING = 0.1
PLOT_SUBJECTS = ['452505', '452506']

# Get subjects
subjects = load_subjects()

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Set figure style
colors, dpi = figure_style()

# Loop over subjects
for i, subject in enumerate(PLOT_SUBJECTS):

    # List sessions
    sessions = os.listdir(join(data_path, 'Subjects', subject))

    # Discard sessions that don't have licks
    no_lick_ses = np.zeros(len(sessions))
    for j, ses in enumerate(sessions):
        if not isfile(join(data_path, 'Subjects', subject, ses, 'lick.positions.npy')):
            no_lick_ses[j] = 1
    sessions = np.array(sessions)[~no_lick_ses.astype(int).astype(bool)]

    # Select final task sessions
    ses_date = [datetime.datetime.strptime(i, '%Y%m%d').date() for i in sessions]
    ses_date = [k for k in ses_date if k >= subjects.loc[subjects['SubjectID'] == subject,
                                                        'DateFinalTask'].values[0]]
    sessions = [datetime.datetime.strftime(i, '%Y%m%d') for i in ses_date]
    if len(sessions) == 0:
        continue

    # Create lick figure
    f, axs = plt.subplots(int(np.ceil(len(sessions)/4)), 4, figsize=(7,  2*np.ceil(len(sessions) / 4)),
                          dpi=dpi)
    if len(sessions) > 4:
        axs = np.concatenate(axs)

    max_y = np.empty(len(sessions))
    for j, ses in enumerate(sessions):

        # Load in data
        trials = pd.read_csv(join(data_path, 'Subjects', subject, ses, 'trials.csv'))
        lick_pos = np.load(join(data_path, 'Subjects', subject, ses, 'lick.positions.npy'))
        
        # Convert to cm
        lick_pos /= 100

        # Get timestamps of entry of goal, no-goal and control object sets
        goal_obj_enters = np.concatenate((
            trials.loc[trials['soundId'] == 1, f'enterObj{subjects.loc[i, "Sound1Obj"]}Pos'],
            trials.loc[trials['soundId'] == 2, f'enterObj{subjects.loc[i, "Sound2Obj"]}Pos']))
        nogoal_obj_enters = np.concatenate((
            trials.loc[trials['soundId'] == 1, f'enterObj{subjects.loc[i, "Sound2Obj"]}Pos'],
            trials.loc[trials['soundId'] == 2, f'enterObj{subjects.loc[i, "Sound1Obj"]}Pos']))
        control_obj_enters = trials[f'enterObj{subjects.loc[i, "ControlObject"]}'].values
        all_obj_enters = np.concatenate((goal_obj_enters, nogoal_obj_enters, control_obj_enters))
        all_obj_ids = np.concatenate(
            (np.ones(goal_obj_enters.shape[0]),
             np.ones(nogoal_obj_enters.shape[0])*2,
             np.ones(control_obj_enters.shape[0])*3))
        all_obj_ids = all_obj_ids[~np.isnan(all_obj_enters)]
        all_obj_enters = all_obj_enters[~np.isnan(all_obj_enters)]

        peri_multiple_events_time_histogram(
            lick_pos, np.ones(lick_pos.shape[0]), all_obj_enters, all_obj_ids,
            [1], t_before=CM_BEFORE, t_after=CM_AFTER, bin_size=BIN_SIZE, smoothing=SMOOTHING, ax=axs[j],
            pethline_kwargs=[{'color': colors['goal'], 'lw': 1},
                             {'color': colors['no-goal'], 'lw': 1},
                             {'color': colors['control'], 'lw': 1}],
            errbar_kwargs=[{'color': colors['goal'], 'alpha': 0.3, 'lw': 0},
                           {'color': colors['no-goal'], 'alpha': 0.3, 'lw': 0},
                           {'color': colors['control'], 'alpha': 0.3, 'lw': 0}],
            raster_kwargs=[{'color': colors['goal'], 'lw': 0.5},
                           {'color': colors['no-goal'], 'lw': 0.5},
                           {'color': colors['control'], 'lw': 0.5}],
            eventline_kwargs={'lw': 0}, include_raster=True)

        axs[j].set(ylabel='Licks/mm', xticks=np.arange(-CM_BEFORE, CM_AFTER+1),
                   yticks=[0, np.ceil(axs[j].get_ylim()[1])],
                   title=f'{ses} ({trials.shape[0]} trials)', xlabel='')
        axs[j].yaxis.set_label_coords(-0.1, 0.75)
        max_y[j] = axs[j].get_ylim()[1]

    # Place the dotted line now we know the y lim extend
    for j, ses in enumerate(sessions):
        axs[j].plot([0, 0], [0, np.ceil(np.max(max_y))], color='grey', ls='--', lw=0.75, zorder=0)

    f.suptitle(f'{subjects.loc[subjects["SubjectID"] == subject, "Nickname"].values[0]} ({subject})')
    f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
    sns.despine(trim=True)
    if int(np.ceil(len(sessions)/4)) > 1:
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, hspace=0.4)
    else:
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.8, hspace=0.4)

    # plt.tight_layout()
    plt.savefig(join(path_dict['fig_path'], f'{subject}_task_licks_pos.jpg'), dpi=600)

