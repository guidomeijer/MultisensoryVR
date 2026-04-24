# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 10/03/2026
"""
# %%

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style, event_aligned_trace
path_dict = paths()
colors, dpi = figure_style()

# Settings
T_BEFORE = 1
T_AFTER = 1
BASELINE = 0.5  # before T_BEFORE
SMOOTHING = 1.2
REGIONS = ['CA1', 'LEC', 'PERI', 'TEa', 'AUD', 'VIS']

# Get data files
assembly_df = pd.read_csv(path_dict['save_path'] / 'assembly_sig.csv', dtype={'subject': str, 'date': str})
activation_files = (path_dict['google_drive_data_path'] / 'Assemblies').rglob('*amplitudes.npy')
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv', dtype={'subject': str, 'date': str})

# Select only significant assemblies
assembly_df = assembly_df[assembly_df['p_ripples'] < 0.05]

# Load in data per region
assembly_act = {'CA1': [], 'LEC': [], 'AUD': [], 'VIS': [], 'TEa': [], 'PERI': []}
for i, file in enumerate(activation_files):
    print(f'Processing {file.stem}')

    # Load in data
    this_act = np.load(file)
    this_region = file.stem.split('.')[0].split('_')[3]
    this_subject = file.stem.split('.')[0].split('_')[0]
    this_date = file.stem.split('.')[0].split('_')[1]
    this_probe = file.stem.split('.')[0].split('_')[2]
    this_time = np.load(file.parent / f'{this_subject}_{this_date}_{this_probe}_{this_region}.times.npy')
    this_ripples = ripples[(ripples['subject'] == this_subject) & (ripples['date'] == this_date)]
    fs = np.round(1 / np.mean(np.diff(this_time)))

    # Select significant assemblies
    these_assemblies = assembly_df[(assembly_df['region'] == this_region)
                                   & (assembly_df['subject'] == this_subject)
                                   & (assembly_df['date'] == this_date)
                                   & (assembly_df['probe'] == this_probe)]['assembly'].values
    if these_assemblies.shape[0] == 0:
        continue
    this_act = this_act[these_assemblies - 1, :]

    # Add some smoothing to this_act
    this_act = gaussian_filter1d(this_act, sigma=SMOOTHING, axis=1)

    # Get mean ripple centered activation
    for j in range(this_act.shape[0]):
        mean_ripple_act = event_aligned_trace(this_act[j, :], this_time, this_ripples['start_times'].values,
                                              t_before=T_BEFORE + BASELINE, t_after=T_AFTER,
                                              baseline=[-T_BEFORE - BASELINE, -T_BEFORE], fs=fs)
        full_time_ax = np.linspace(-(T_BEFORE + BASELINE), T_AFTER, mean_ripple_act.shape[0])
        mean_ripple_act = mean_ripple_act[full_time_ax >= -1]
        assembly_act[this_region].append(mean_ripple_act)

time_ax = full_time_ax[full_time_ax >= -1]

# %% Get average assembly activation for each region
assembly_mean = np.zeros((len(REGIONS), len(time_ax)))
assembly_norm = {'CA1': [], 'LEC': [], 'AUD': [], 'VIS': [], 'TEa': [], 'PERI': []}
for r, region in enumerate(REGIONS):
    assembly_act[region] = np.vstack(assembly_act[region])
    assembly_norm[region] = np.vstack(assembly_act[region])

    # Sort assemblies by their max activation and normalize
    for ii in range(assembly_act[region].shape[0]):
        if np.abs(np.max(assembly_act[region][ii, :])) > np.abs(np.min(assembly_act[region][ii, :])):
            assembly_norm[region][ii, :] = assembly_act[region][ii, :] / np.max(assembly_act[region][ii, :])
        else:
            assembly_norm[region][ii, :] = assembly_act[region][ii, :] / np.abs(np.min(assembly_act[region][ii, :]))
    sort_idx = np.argsort(np.max(assembly_norm[region], axis=1))
    assembly_norm[region] = assembly_norm[region][sort_idx, :]

    # Take the mean over assemblies per region
    assembly_mean[r, :] = np.mean(assembly_act[region], axis=0)
    assembly_mean[r, :] = assembly_mean[r, :] / np.max(assembly_mean[r, :])

# %% Plot means
f, ax1 = plt.subplots(figsize=(1.5, 1.75), dpi=dpi)
imh = ax1.imshow(assembly_mean, aspect='auto', cmap='coolwarm', clim=[-1, 1])
ax1.set(yticks=np.arange(len(REGIONS)), yticklabels=REGIONS, xticks=[0, 20, 40], xticklabels=[-1, 0, 1],
        xlabel='Time from ripple onset (s)')
cbar = plt.colorbar(imh, ax=ax1)
cbar.set_label('Assembly activation', rotation=270, labelpad=8)

plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'mean_ripple_assemblies.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'mean_ripple_assemblies.pdf')
plt.show()

# %% Plot indiviual assemblies per region
f, axs = plt.subplots(1, len(REGIONS), figsize=(1.75 * len(REGIONS), 1.75), dpi=dpi)
for r, region in enumerate(REGIONS):
    #imh = axs[r].imshow(assembly_act[region], aspect='auto', cmap='coolwarm', clim=[-1, 1])
    imh = axs[r].imshow(assembly_norm[region], aspect='auto', cmap='coolwarm', clim=[-1, 1])
    axs[r].set(xticks=[0, 20, 40], xticklabels=[-1, 0, 1], yticks=[], title=region)
    #axs[r].invert_yaxis()
    if r == len(REGIONS):
        cbar = plt.colorbar(imh, ax=axs[r])

plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'ripple_assemblies.jpg', dpi=600)
plt.show()