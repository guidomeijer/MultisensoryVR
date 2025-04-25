# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:28:35 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from msvr_functions import paths, load_subjects, figure_style, combine_regions
colors, dpi = figure_style()

# Load in data
path_dict = paths()
with open(join(path_dict['save_path'], 'env_act_dict.pkl'), 'rb') as f:
    env_act_dict = pickle.load(f)
    
# Loop over regions
distance_region = dict()
for i, region in enumerate(env_act_dict.keys()):
    if (region == 'root') | (region == 'ENT'):
        continue
    
    # this_act: shape (neurons, distance_bins, context) → (415, 150, 2)
    this_act = env_act_dict[region]  
    n_bins = this_act.shape[1]
    distances = np.zeros(n_bins)
    
    for i in range(n_bins):
        vec1 = this_act[:, i, 0]  # shape: (neurons,)
        vec2 = this_act[:, i, 1]  # shape: (neurons,)
        distances[i] = np.linalg.norm(vec1 - vec2)
    
    distance_region[region] = distances

# Order regions by effect
sorted_regions = sorted(distance_region.keys(), key=lambda k: np.max(distance_region[k]))
    
# %% Plot

f, ax = plt.subplots(1, 1, figsize=(2, 4.5), dpi=dpi)
offset = 0
for i, region in enumerate(sorted_regions):
    ax.plot(distance_region[region] + offset, label=region, color=colors[region])
    ax.plot([0, 150], [offset, offset], color='grey', ls='--')
    ax.text(155, offset, region, ha='left', va='bottom', color=colors[region])
    offset += np.max(distance_region[region]) * 1.4
ax.plot([25, 25], ax.get_ylim())
    
ax.axis('off')
plt.tight_layout()
    
