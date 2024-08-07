# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:29:49 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style
colors, dpi = figure_style()

# Load in data
path_dict = paths()
stats_df = pd.read_csv(join(path_dict['save_path'], 'significant_neurons.csv'))
stats_df = stats_df[stats_df['region'] != 'root']

# Get percentage of significant neurons per region
region_df = stats_df[['region', 'sig_goal', 'sig_obj_onset', 'sig_control', 'sig_obj_diff']].groupby('region').sum()
region_df['n_neurons'] = stats_df.groupby('region').size()
region_df['perc_goal'] = (region_df['sig_goal'] / region_df['n_neurons']) * 100
region_df['perc_obj_onset'] = (region_df['sig_obj_onset'] / region_df['n_neurons']) * 100
region_df['perc_control'] = (region_df['sig_control'] / region_df['n_neurons']) * 100
region_df['perc_obj_diff'] = (region_df['sig_obj_diff'] / region_df['n_neurons']) * 100
region_df = region_df.reset_index()

long_df = pd.melt(region_df, id_vars=['region'],
                  value_vars=['perc_goal', 'perc_obj_onset', 'perc_control', 'perc_obj_diff'])

# Plot
f, ax = plt.subplots(1, 1, figsize=(7, 1.75), dpi=dpi) 
sns.barplot(data=long_df, x='variable', y='value', hue='region', ax=ax)
