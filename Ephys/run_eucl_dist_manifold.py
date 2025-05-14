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
with open(join(path_dict['save_path'], 'env_act_dict_1_3.pkl'), 'rb') as f:
    env_act_dict = pickle.load(f)
    
# Loop over regions
distance_region = dict()
for i, region in enumerate(env_act_dict.keys()):
    if region == 'position':
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
    ax.plot(env_act_dict['position'], distance_region[region] + offset, label=region, color=colors[region])
    ax.plot([env_act_dict['position'][0], env_act_dict['position'][-1]], [offset, offset], color='grey', ls='--')
    ax.text(env_act_dict['position'][-1] + 5, offset, region, ha='left', va='bottom', color=colors[region])
    offset += np.max(distance_region[region]) * 1.4
vert_line_y = ax.get_ylim()
ax.plot([2.5, 2.5], vert_line_y, color='k', lw=0.5)
ax.plot([45, 45], vert_line_y, color='k', lw=0.5)
ax.plot([90, 90], vert_line_y, color='k', lw=0.5)
ax.plot([135, 135], vert_line_y, color='k', lw=0.5)
    
#ax.axis('off')
plt.tight_layout()
plt.savefig(join(path_dict['google_drive_fig_path'], 'eu_dist_context.jpg'), dpi=600)
    
