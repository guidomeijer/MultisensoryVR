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
with open(join(path_dict['local_data_path'], 'environment_activity_1-3.pkl'), 'rb') as f:
    env_act_dict = pickle.load(f)
with open(join(path_dict['local_data_path'], 'environment_activity_shuf_1-3.pkl'), 'rb') as f:
    env_act_dict_shuf = pickle.load(f)
with open(join(path_dict['local_data_path'], 'environment_info_1-3.pkl'), 'rb') as f:
    env_info = pickle.load(f)
    
# %% All neurons together

# Loop over regions
distance_region, dist_shuf_region = dict(), dict()
for region in ['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'dCA1', 'vCA1']:
    
    # Calculate Eucledian distance
    n_bins = env_act_dict[region].shape[1]
    n_shuffles = env_act_dict_shuf[region].shape[3]
    distances = np.zeros(n_bins)
    dist_shuf = np.zeros((n_shuffles, n_bins))
    for ii in range(n_bins):
        distances[ii] = np.linalg.norm(env_act_dict[region][:, ii, 0] - env_act_dict[region][:, ii, 1])
        for jj in range(n_shuffles):
            dist_shuf[jj, ii] = np.linalg.norm(env_act_dict_shuf[region][:, ii, 0, jj]
                                              - env_act_dict_shuf[region][:, ii, 1, jj])
    distance_region[region] = distances
    dist_shuf_region[region] = dist_shuf
  
# Plot each region in a separate subplot
n_regions = len(distance_region)
fig, axes = plt.subplots(2, 4, figsize=(8.5, 3), dpi=dpi)
axes = np.concatenate(axes)
x = env_info['position']

for idx, (region, distances) in enumerate(distance_region.items()):
    ax = axes[idx]
    ax.plot(x, distances, color=colors[region], label=region)
    lower = np.percentile(dist_shuf_region[region], 2.5, axis=0)
    upper = np.percentile(dist_shuf_region[region], 97.5, axis=0)
    ax.fill_between(x, lower, upper, color='grey', alpha=0.3, lw=0)
    line_height = ax.get_ylim()
    ax.plot([45, 45], line_height, color='grey', lw=0.5, ls='--')
    ax.plot([90, 90], line_height, color='grey', lw=0.5, ls='--')
    ax.plot([135, 135], line_height, color='grey', lw=0.5, ls='--')
    ax.plot([-12, -12], [0, 2], color='k')
    ax.set(title=region, xticks=np.arange(0, 151, 25), yticks=[])
    if idx in [0, 1, 2]:
        ax.set(xticklabels=[])

axes[0].text(-18, 1, '2 E.D.', rotation=90, ha='center', va='center')
axes[-1].axis('off')
sns.despine(trim=True, left=True)
fig.supxlabel('Position (cm)', fontsize=7, y=0.04)
plt.tight_layout()
plt.savefig(join(path_dict['google_drive_fig_path'], 'eucl_dist_context.jpg'), dpi=600)


# %% Split by recording

# Loop over regions
distance_region = dict()
dist_shuf_region = dict()
for region in ['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'dCA1', 'vCA1']:
    
    n_bins = env_act_dict[region].shape[1]
    n_shuffles = env_act_dict_shuf[region].shape[3]
    n_subjects = np.unique(env_info[f'{region}_subject']).shape[0]
    distances = np.empty((n_subjects, n_bins))
    dist_shuf = np.empty((n_subjects, n_shuffles, n_bins))
    
    for s, sub in enumerate(np.unique(env_info[f'{region}_subject'])):
        
        # Calculate Eucledian distance
        for ii in range(n_bins):
            distances[s, ii] = np.linalg.norm(
                env_act_dict[region][env_info[f'{region}_subject'] == sub, ii, 0]
                - env_act_dict[region][env_info[f'{region}_subject'] == sub, ii, 1]
                )
            
            for jj in range(n_shuffles):
                dist_shuf[s, jj, ii] = np.linalg.norm(
                    env_act_dict_shuf[region][env_info[f'{region}_subject'] == sub, ii, 0, jj]
                    - env_act_dict_shuf[region][env_info[f'{region}_subject'] == sub, ii, 1, jj]
                    )
            
        distance_region[region] = distances
        dist_shuf_region[region] = dist_shuf
  
# %% Plot each subject in a separate subplot
for region in ['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'dCA1', 'vCA1']:
    n_subjects = distance_region[region].shape[0]
    fig, axes = plt.subplots(1, n_subjects, figsize=(2*n_subjects, 2), dpi=dpi)
    if n_subjects == 1:
        axes = [axes]
    x = env_info['position']
    
    for idx in range(n_subjects):
        ax = axes[idx]
        ax.plot(x, distance_region[region][idx, :], color=colors[region], label=region)
        lower = np.percentile(dist_shuf_region[region][idx, :, :], 2.5, axis=0)
        upper = np.percentile(dist_shuf_region[region][idx, :, :], 97.5, axis=0)
        ax.fill_between(x, lower, upper, color='grey', alpha=0.3, lw=0)
        line_height = ax.get_ylim()
        ax.plot([45, 45], line_height, color='grey', lw=0.5, ls='--')
        ax.plot([90, 90], line_height, color='grey', lw=0.5, ls='--')
        ax.plot([135, 135], line_height, color='grey', lw=0.5, ls='--')
        ax.plot([-12, -12], [0, 2], color='k')
        ax.set(xticks=np.arange(0, 151, 25), yticks=[])
        if idx in [0, 1, 2]:
            ax.set(xticklabels=[])
    
    axes[0].text(-18, 1, '2 E.D.', rotation=90, ha='center', va='center')
    axes[-1].axis('off')
    sns.despine(trim=True, left=True)
    fig.supxlabel('Position (cm)', fontsize=7, y=0.04)
    plt.tight_layout()

