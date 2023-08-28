# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:43:03 2023

By Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk
import matplotlib
from matplotlib import colors as matplotlib_colors
import json
from os.path import join, realpath, dirname, isfile


def paths():
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input
    
    Save directory can be either the repository (for small files) or the server for large files
    
    Input
    ------------------------
    save_dir : str
        'repo' or 'server' for saving in the repository or one cache, respectively
        
    Output
    ------------------------
    path_dict : dictionary
        Dict with the paths
    """
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        path_dict = dict()
        path_dict['fig_path'] = input('Path folder to save figures: ')
        path_dict['server_path'] = input('Path folder to server: ')
        path_dict['save_path'] = join(dirname(realpath(__file__)), 'Data')
        path_dict['repo_path'] = dirname(realpath(__file__))
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(path_dict, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        path_dict = json.load(json_file)
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


