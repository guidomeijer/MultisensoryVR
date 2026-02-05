# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:20:19 2025 by Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style, combine_regions, load_neural_data
colors, dpi = figure_style()

# Load in data
path_dict = paths()
neurons_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
neurons_df = neurons_df[neurons_df['region'] == 'PERI 36']

# Get mean and sem per recording
mean_df = neurons_df[['date', 'z_goal', 'y']].groupby('date').mean()
sem_df = neurons_df[['date', 'z_goal', 'y']].groupby('date').sem()

# Get percentage significant per recording
perc_sig_df = neurons_df[['date']].groupby('date').size().to_frame()
perc_sig_df['sig_neurons'] = neurons_df[['date', 'sig_goal']].groupby('date').sum()['sig_goal']
perc_sig_df['y'] = neurons_df[['date', 'y']].groupby('date').mean()['y']
perc_sig_df['perc_sig'] = (perc_sig_df['sig_neurons'] / perc_sig_df[0]) * 100

# Plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
ax1.errorbar(mean_df['y'], mean_df['z_goal'],
             xerr=sem_df['y'], yerr=sem_df['z_goal'],
             fmt='none', ecolor=[0.7, 0.7, 0.7], capsize=2, capthick=1, zorder=0)
ax1.scatter(mean_df['y'], mean_df['z_goal'], marker='s', color='k', s=20, zorder=1)

ax2.scatter(perc_sig_df['y'], perc_sig_df['perc_sig'])