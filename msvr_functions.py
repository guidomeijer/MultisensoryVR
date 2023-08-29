# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:43:03 2023

By Guido Meijer
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
import json
import shutil
import datetime
from os.path import join, realpath, dirname, isfile, split, isdir


def paths(synchronize=True):
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input

    Save directory can be either the repository (for small files) or the server for large files

    This function also runs the synchronization between the server and the local data folder

    Input
    ------------------------
    save_dir : str
        'repo' or 'server' for saving in the repository or one cache, respectively

    Output
    ------------------------
    path_dict : dictionary
        Dict with the paths
    """

    # Get the paths
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        path_dict = dict()
        path_dict['fig_path'] = input('Path folder to save figures: ')
        path_dict['server_path'] = input('Path folder to server: ')
        path_dict['local_data_path'] = input('Path folder to local data: ')
        path_dict['save_path'] = join(dirname(realpath(__file__)), 'Data')
        path_dict['repo_path'] = dirname(realpath(__file__))
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(path_dict, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        path_dict = json.load(json_file)

    # Synchronize data from the server with the local data folder
    if synchronize & isdir(path_dict['server_path']):

        # Create Subjects folder if it doesn't exist
        if not isdir(join(path_dict['local_data_path'], 'Subjects')):
            os.mkdir(join(path_dict['local_data_path'], 'Subjects'))

        # Read in the time of last sync
        if isfile(join(path_dict['local_data_path'], 'sync_timestamp.txt')):
            f = open(join(path_dict['local_data_path'], 'sync_timestamp.txt'), 'r')
            sync_date = datetime.datetime.strptime(f.read(), '%Y%m%d').date()
            f.close()
        else:
            # If never been synched set date to yesterday so that it runs the first time
            sync_date = datetime.date.today() - datetime.timedelta(days=1)

        # Synchronize server with local data once a day
        if (datetime.date.today() - sync_date).days > 0:
            print('Synchronizing data from server with local data folder')

            # Copy data from server to local folder
            subjects = os.listdir(join(path_dict['server_path'], 'Subjects'))
            for i, subject in enumerate(subjects):
                if not isdir(join(path_dict['local_data_path'], 'Subjects', subject)):
                    os.mkdir(join(path_dict['local_data_path'], 'Subjects', subject))
                sessions = os.listdir(join(path_dict['server_path'], 'Subjects', subject))
                for j, session in enumerate(sessions):
                    print(
                        f'Copying {join(path_dict["server_path"], "Subjects", subject, session)}')
                    if not isdir(join(path_dict['local_data_path'], 'Subjects', subject, session)):
                        os.mkdir(join(path_dict['local_data_path'], 'Subjects', subject, session))
                    files = [f for f in os.listdir(join(path_dict['server_path'], 'Subjects', subject, session))
                             if isfile(join(path_dict['server_path'], 'Subjects', subject, session, f))]

                    for f, file in enumerate(files):
                        if not isfile(join(path_dict['local_data_path'], 'Subjects', subject, session, file)):
                            shutil.copyfile(join(path_dict['server_path'], 'Subjects', subject, session, file),
                                            join(path_dict['local_data_path'], 'Subjects', subject, session, file))

            # Update synchronization timestamp
            with open(join(path_dict['local_data_path'], 'sync_timestamp.txt'), 'w') as f:
                f.write(datetime.date.today().strftime('%Y%m%d'))
                f.close()
            print('Done')

    return path_dict


def figure_style():
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 7,
                "figure.titlesize": 7,
                "axes.titlesize": 7,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                'legend.fontsize': 7,
                'legend.title_fontsize': 7,
                'legend.frameon': False,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    colors = {}

    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 10
    return colors, dpi


def load_subjects():
    path_dict = paths()
    subjects = pd.read_csv(join(path_dict['repo_path'], 'subjects.csv'),
                           delimiter=';|,',
                           engine='python')
    subjects['SubjectID'] = subjects['SubjectID'].astype(str)
    return subjects


def peri_event_trace(array, timestamps, event_times, event_ids, ax, t_before=1, t_after=3,
                     event_labels=None, color_palette='colorblind', ind_lines=False, kwargs=[]):

    # Construct dataframe for plotting
    plot_df = pd.DataFrame()
    time_x = np.arange(-t_before + np.diff(timestamps)[0]/2, t_after + np.diff(timestamps)[0]/2,
                       np.diff(timestamps)[0])
    for i, t in enumerate(event_times[~np.isnan(event_times)]):
        plot_df = pd.concat((plot_df, pd.DataFrame(data={
            'y': array[(timestamps >= t-t_before) & (timestamps < t+t_after)],
            'time': time_x, 'event_id': event_ids[i], 'event_nr': i})))

    # Plot
    if ind_lines:
        sns.lineplot(data=plot_df, x='time', y='y', hue='event_id', estimator=None, units='event_nr',
                     palette=color_palette, err_kws={'lw': 0}, ax=ax, **kwargs)
    else:
        sns.lineplot(data=plot_df, x='time', y='y', hue='event_id', errorbar='se',
                     palette=color_palette, err_kws={'lw': 0}, ax=ax, **kwargs)
    if event_labels is None:
        ax.get_legend().remove()
    else:
        g = ax.legend(title='', prop={'size': 5.5})
        for t, l in zip(g.texts, event_labels):
            t.set_text(l)

    # sns.despine(trim=True)
    # plt.tight_layout()
