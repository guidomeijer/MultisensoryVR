# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:05:33 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style
colors, dpi = figure_style()

# Load in data
path_dict = paths()
corr_df = pd.read_pickle(join(path_dict['google_drive_data_path'], 'corr_mua.pickle'))
corr_df['region_pair'] = [corr_df.loc[i, 'region1'] + '-' + corr_df.loc[i, 'region2'] for i in corr_df.index]
region_pairs = corr_df['region_pair'].unique()
regions = np.array(['AUD', 'VIS', 'TEa', 'PERI', 'LEC', 'CA1'])

# Get diagonal
diag_df = pd.DataFrame()
for i, region_pair in enumerate(region_pairs):
    for obj in [1, 2]:
        this_df = corr_df[(corr_df['region_pair'] == region_pair) & (corr_df['object'] == obj)]
        corr_arr = np.stack(this_df['corr_matrix'].values)

        for k in range(corr_arr.shape[0]):
            this_corr = np.squeeze(corr_arr[k, :, :])

            # Get the average of the diagonal for each column plus minus 2 above and below the diagonal
            n = this_corr.shape[0]
            diag_means = []
            for col in range(n):
                row_start = max(0, col - 3)
                row_end = min(n, col + 4)
                diag_means.append(np.mean(this_corr[row_start:row_end, col]))
            diag_means = np.array(diag_means)

            # Do baseline subtraction
            time_ax = corr_df['time_ax'].values[0]
            #diag_means = diag_means - np.mean(diag_means[time_ax < 0])

            # Get asymetry around the diagonal
            diag_asym = []
            for col in range(n):
                row_start = max(0, col - 10)
                row_end = min(n, col + 10)
                diag_asym.append(np.mean(this_corr[col+1:row_end, col]) - np.mean(this_corr[row_start:col, col]))
            diag_asym = np.array(diag_asym)

            # Add to dataframe
            diag_df = pd.concat((diag_df, pd.DataFrame(data={
                'corr': diag_means, 'asym': diag_asym, 'region_pair': region_pair, 'time_ax': time_ax, 'object': obj
            })), ignore_index=True)


# %% Plot
cmap_colors = ["#0055ff", "black", "#ff0000", "#ffff00"]
nodes = [0.0, 0.5, 0.8, 1.0]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("diverging_black", list(zip(nodes, cmap_colors)))
plot_region1 = np.array(['TEa', 'PERI', 'LEC', 'CA1', 'AUD'])
plot_region2 = np.array(['VIS', 'TEa', 'PERI', 'LEC', 'CA1'])
time_min = -1
time_max = 1

f, axs = plt.subplots(5, 5, figsize=(5, 4), dpi=dpi, sharex=True, sharey=True)
for i, region1 in enumerate(plot_region1):
    for j, region2 in enumerate(plot_region2):
        this_df = corr_df[(corr_df['region1'] == region1) & (corr_df['region2'] == region2) & (corr_df['object'] == 1)]
        if len(this_df) == 0:
            axs[i,j].axis('off')
            continue
        avg_corr = np.mean(np.stack(this_df['corr_matrix'].values), axis=0)
        axs[i,j].imshow(avg_corr, cmap=custom_cmap, aspect='auto', extent=(time_min, time_max, time_max, time_min),
                        clim=(-0.5, 0.5))
        axs[i,j].invert_yaxis()
        axs[i,j].plot([0, 0], [time_min, time_max], color='white', lw=0.5, ls='--')
        axs[i,j].plot([time_min, time_max], [0, 0], color='white', lw=0.5, ls='--')
        if j == 0:
            axs[i,j].set(ylabel=region1, xticks=[time_min, 0, time_max], yticks=[time_min, 0, time_max],
                         yticklabels=[time_min, 0, time_max], xticklabels=[time_min, 0, time_max])
        if i == 4:
            axs[i,j].set(xlabel=region2, xticks=[time_min, 0, time_max], yticks=[time_min, 0, time_max],
                         yticklabels=[time_min, 0, time_max], xticklabels=[time_min, 0, time_max])
f.suptitle('Object 1')

sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %%
f, axs = plt.subplots(5, 5, figsize=(5, 4), dpi=dpi, sharex=True, sharey=True)
for i, region1 in enumerate(plot_region1):
    for j, region2 in enumerate(plot_region2):
        this_df = corr_df[(corr_df['region1'] == region1) & (corr_df['region2'] == region2) & (corr_df['object'] == 2)]
        if len(this_df) == 0:
            axs[i,j].axis('off')
            continue
        avg_corr = np.mean(np.stack(this_df['corr_matrix'].values), axis=0)
        axs[i,j].imshow(avg_corr, cmap=custom_cmap, aspect='auto', extent=(time_min, time_max, time_max, time_min),
                        clim=(-0.5, 0.5))
        axs[i,j].invert_yaxis()
        axs[i,j].plot([0, 0], [time_min, time_max], color='white', lw=0.5, ls='--')
        axs[i,j].plot([time_min, time_max], [0, 0], color='white', lw=0.5, ls='--')
        if j == 0:
            axs[i,j].set(ylabel=region1, xticks=[time_min, 0, time_max], yticks=[time_min, 0, time_max],
                         yticklabels=[time_min, 0, time_max], xticklabels=[time_min, 0, time_max])
        if i == 4:
            axs[i,j].set(xlabel=region2, xticks=[time_min, 0, time_max], yticks=[time_min, 0, time_max],
                         yticklabels=[time_min, 0, time_max], xticklabels=[time_min, 0, time_max])

f.suptitle('Object 2')

sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %%
f, axs = plt.subplots(3, 5, figsize=(7, 4), dpi=dpi, sharex=True)
axs = axs.flatten()
for i, region_pair in enumerate(region_pairs):
    plot_df = diag_df[diag_df['region_pair'] == region_pair]
    sns.lineplot(plot_df, x='time_ax', y='corr', ax=axs[i], errorbar='se', hue='object', hue_order=[1, 2],
                 palette=[colors['obj1'], colors['obj2']], err_kws={'lw': 0}, legend=None)
    axs[i].set(title=f'{region_pair}', ylabel='', xlabel='')

axs[0].set(ylabel='Correlation (r)')

sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %%
f, axs = plt.subplots(3, 5, figsize=(7, 4), dpi=dpi, sharex=True, sharey=True)
axs = axs.flatten()
for i, region_pair in enumerate(region_pairs):
    plot_df = diag_df[diag_df['region_pair'] == region_pair]
    sns.lineplot(plot_df, x='time_ax', y='asym', ax=axs[i], errorbar='se', hue='object', hue_order=[1, 2],
                 palette=[colors['obj1'], colors['obj2']], err_kws={'lw': 0}, legend=None)
    axs[i].plot([-0.5, 0.5], [0, 0], color='grey', ls='--', lw=0.5)
    axs[i].set(title=f'{region_pair}', ylabel='', xlabel='', ylim=[-0.2, 0.2])

axs[0].set(ylabel='Asymmetry (r)')

sns.despine(trim=True)
plt.tight_layout()
plt.show()