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
with open(join(path_dict['local_data_path'], 'env_act_dict_1-3.pkl'), 'rb') as f:
    env_act_dict = pickle.load(f)
with open(join(path_dict['local_data_path'], 'env_act_dict_shuf_1-3.pkl'), 'rb') as f:
    env_act_dict_shuf = pickle.load(f)
    
# Loop over regions
distance_region, dist_shuf_region = dict(), dict()
for i, region in enumerate(env_act_dict.keys()):
    if region == 'position':
        continue
    
    # Calculate Eucledian distance
    n_bins = env_act_dict[region].shape[1]
    n_shuffles = env_act_dict_shuf[region].shape[3]
    distances = np.zeros(n_bins)
    dist_shuf = np.zeros((n_shuffles, n_bins))
    for i in range(n_bins):
        distances[i] = np.linalg.norm(env_act_dict[region][:, i, 0] - env_act_dict[region][:, i, 1])
        for j in range(n_shuffles):
            dist_shuf[j, i] = np.linalg.norm(env_act_dict_shuf[region][:, i, 0, j]
                                             - env_act_dict_shuf[region][:, i, 1, j])
    distance_region[region] = distances
    dist_shuf_region[region] = dist_shuf
    

# %% Plot each region in a separate subplot
n_regions = len(distance_region)
fig, axes = plt.subplots(n_regions, 1, figsize=(6, 3 * n_regions), dpi=dpi, sharex=True)

x = env_act_dict['position']

for idx, (region, distances) in enumerate(distance_region.items()):
    
    ax = axes[idx]
    # Plot main line
    ax.plot(x, distances, color=colors[region], label=region)
    # Compute 95% CI for null distribution
    lower = np.percentile(dist_shuf_region[region], 2.5, axis=0)
    upper = np.percentile(dist_shuf_region[region], 97.5, axis=0)
    ax.fill_between(x, lower, upper, color='grey', alpha=0.3, label='95% CI (null)')
    ax.set_ylabel('Eucledian distance')
    ax.set_title(region)
    ax.legend()

axes[-1].set_xlabel('Position')
plt.tight_layout()

"""
# Order regions by effect
sorted_regions = sorted(distance_region.keys(), key=lambda k: np.max(distance_region[k]))
    
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
"""    
