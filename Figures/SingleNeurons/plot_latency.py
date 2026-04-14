# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 14/04/2026
"""
# %%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style
path_dict = paths()
colors, dpi = figure_style()

# Load in data
latency_df = pd.read_csv(path_dict['save_path'] / 'latency_goal_nogoal_obj1_neurons.csv')
modulation_df = pd.read_csv(path_dict['save_path'] / 'significant_neurons.csv')
merged_df = pd.merge(latency_df, modulation_df, on=['subject', 'date', 'probe', 'neuron_id', 'region', 'allen_acronym'])

# Select significant neurons
merged_df = merged_df[(merged_df['p_obj_onset'] < 0.05) & (merged_df['region'] != 'root')]

# Get order
region_order = merged_df[['region', 'latency']].groupby('region').mean().sort_values('latency').index.values

# Plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.75*2, 1.75), dpi=dpi)

sns.barplot(merged_df, x='region', y='latency', order=region_order, ax=ax1, errorbar='se')


plt.show()