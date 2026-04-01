# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 10/03/2026
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style
path_dict = paths()
colors, dpi = figure_style()

# Region order
REGIONS = ['CA1', 'LEC', 'PERI', 'TEa', 'AUD', 'VIS']

# Get data files
assembly_df = pd.read_csv(path_dict['save_path'] / 'assembly_sig.csv')
assembly_df['subject'] = assembly_df['subject'].astype(str)
assembly_df['date'] = assembly_df['date'].astype(str)
assembly_df['probe'] = assembly_df['probe'].astype(str)
activation_files = (path_dict['google_drive_data_path'] / 'RippleAssemblies').rglob('*mean_activation.npy')
time_files = (path_dict['google_drive_data_path'] / 'RippleAssemblies').rglob('*time.npy')
time_ax = np.load(list(time_files)[0])

# Select only significant assemblies
assembly_df = assembly_df[assembly_df['p_ripples'] < 0.05]

# Load in data per region
assembly_act = {'CA1': [], 'LEC': [], 'AUD': [], 'VIS': [], 'TEa': [], 'PERI': []}
for i, file in enumerate(activation_files):

    # Load in data
    this_act = np.load(file)

    # Normalize assemblies
    for ii in range(this_act.shape[0]):
        this_act[ii, :] = this_act[ii, :] - np.mean(this_act[ii, time_ax < -0.5])

    # Get info
    this_region = file.stem.split('_')[3]
    this_subject = file.stem.split('_')[0]
    this_date = file.stem.split('_')[1]
    this_probe = file.stem.split('_')[2]

    # Select significant assemblies
    these_assemblies = assembly_df[(assembly_df['region'] == this_region)
                                   & (assembly_df['subject'] == this_subject)
                                   & (assembly_df['date'] == this_date)
                                   & (assembly_df['probe'] == this_probe)]['assembly'].values
    assembly_act[this_region].append(this_act[these_assemblies - 1, :])

# Get average assembly activation for each region
assembly_mean = np.zeros((len(REGIONS), len(time_ax)))
assembly_norm = {'CA1': [], 'LEC': [], 'AUD': [], 'VIS': [], 'TEa': [], 'PERI': []}
for r, region in enumerate(REGIONS):
    assembly_act[region] = np.vstack(assembly_act[region])

    # Sort assemblies by their max activation and normalize
    sort_idx = np.argsort(np.max(assembly_act[region], axis=1))
    assembly_norm[region] = assembly_act[region][sort_idx, :]
    for ii in range(assembly_norm[region].shape[0]):
        if np.abs(np.max(assembly_norm[region][ii, :])) > np.abs(np.min(assembly_norm[region][ii, :])):
            assembly_norm[region][ii, :] = assembly_norm[region][ii, :] / np.max(assembly_norm[region][ii, :])
        else:
            assembly_norm[region][ii, :] = assembly_norm[region][ii, :] / np.abs(np.min(assembly_norm[region][ii, :]))

    # Take the mean over assemblies per region
    assembly_mean[r, :] = np.mean(assembly_act[region], axis=0)
    assembly_mean[r, :] = assembly_mean[r, :] / np.max(assembly_mean[r, :])

# %% Plot means
f, ax1 = plt.subplots(figsize=(1.5, 1.75), dpi=dpi)
imh = ax1.imshow(assembly_mean, aspect='auto', cmap='coolwarm')
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
    imh = axs[r].imshow(assembly_norm[region], aspect='auto', cmap='coolwarm')
    axs[r].set(xticks=[0, 20, 40], xticklabels=[-1, 0, 1], yticks=[], title=region)
    #axs[r].invert_yaxis()
    if r == len(REGIONS):
        cbar = plt.colorbar(imh, ax=axs[r])

plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / 'ripple_assemblies.jpg', dpi=600)
plt.show()