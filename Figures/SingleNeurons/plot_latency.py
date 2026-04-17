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
merged_df = merged_df[merged_df['region'] != 'root']
#merged_df = merged_df[(merged_df['p_reward'] < 0.05) & (merged_df['region'] != 'root')]

# Get order
region_order = merged_df[['region', 'latency']].groupby('region').mean().sort_values('latency').index.values

# %% Plot
f, ax1 = plt.subplots(1, 1, figsize=(1.4, 1.75), dpi=dpi)

sns.barplot(merged_df, x='region', y='latency', order=region_order, ax=ax1, errorbar='se', hue_order=region_order,
            hue='region', palette=colors)
ax1.set(ylabel='Reward modulation latency (s)', xlabel='', yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        yticklabels=[0, 0.2, 0.4, 0.6, 0.8, 1])
ax1.tick_params(axis='x', labelrotation=90)

sns.despine(trim=False)
plt.tight_layout()

plt.show()