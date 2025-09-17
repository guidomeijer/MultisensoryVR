# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 14:47:37 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from msvr_functions import paths, figure_style
colors, dpi = figure_style()
path_dict = paths()

"""
waveform_df = pd.read_csv(path_dict['save_path'] / 'waveform_metrics.csv')
waveform_df['spike_width'] = waveform_df['peak_to_valley'] * 1000
waveform_df['neuron_type'] = 'Und.'
waveform_df.loc[waveform_df['spike_width'] < 0.45, 'neuron_type'] = 'INT'
waveform_df.loc[waveform_df['spike_width'] > 0.45, 'neuron_type'] = 'PYR'
waveform_df.to_csv(path_dict['save_path'] / 'waveform_metrics.csv', index=False)
waveform_df = waveform_df[waveform_df['good'] == 1]
waveform_df['recovery_slope'] = waveform_df['recovery_slope'] / 1000
waveform_df['repolarization_slope'] = waveform_df['repolarization_slope'] / 1000
"""


waveform_df = pd.read_csv(path_dict['save_path'] / 'waveform_metrics_calc.csv')
waveform_df.loc[waveform_df['spike_width'] >= 0.4, 'neuron_type'] = 'PYR'
waveform_df.loc[waveform_df['spike_width'] < 0.4, 'neuron_type'] = 'INT'
waveform_df.loc[(waveform_df['region'] == 'PERI 36') & (waveform_df['spike_width'] < 0.55),
                'neuron_type'] = 'INT'
waveform_df.loc[(waveform_df['region'] == 'PERI 36') & (waveform_df['spike_width'] >= 0.55),
                'neuron_type'] = 'PYR'
waveform_df.loc[(waveform_df['region'] == 'PERI 35') & (waveform_df['spike_width'] < 0.55),
                'neuron_type'] = 'INT'
waveform_df.loc[(waveform_df['region'] == 'PERI 35') & (waveform_df['spike_width'] >= 0.55),
                'neuron_type'] = 'PYR'
waveform_df.to_csv(path_dict['save_path'] / 'waveform_metrics.csv', index=False)
waveform_df = waveform_df[waveform_df['good'] == 1]
waveform_df['peak_trough_ratio'] = waveform_df['pt_ratio']
waveform_df['recovery_slope'] = waveform_df['rc_slope']
waveform_df['repolarization_slope'] = waveform_df['rp_slope']

# Print the percentages
for region in np.unique(waveform_df['region']):
    region_n = waveform_df[waveform_df['region'] == region].groupby('neuron_type').size()
    print(f'{region}: {np.round((region_n["INT"] / region_n["PYR"]) * 100)}% NS neurons'
          f' out of total {region_n["INT"] + region_n["PYR"]}')

# %% Plot spike with per region
regions = np.unique(waveform_df['region'])

f, axs = plt.subplots(2, 4, figsize=(8, 4), dpi=dpi, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(regions):
    if region == 'root':
        continue
    sns.histplot(data=waveform_df[waveform_df['region'] == region], x='spike_width', ax=axs[i],
                 binwidth=0.032)
    axs[i].set(title=region, xlim=[0, 1.5], ylabel='', xlabel='')
f.suptitle('Spike width (ms)')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'WaveformClassification' / 'spike_width.jpg',
            dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(8, 4), dpi=dpi, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(regions):
    if region == 'root':
        continue
    sns.histplot(data=waveform_df[waveform_df['region'] == region], x='peak_trough_ratio', ax=axs[i],
                 binwidth=0.02)
    axs[i].set(title=region, xlim=[-1, 0], ylabel='', xlabel='')
f.suptitle('Peak-to-trough ratio')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'WaveformClassification' / 'peak_trough_ratio.jpg',
            dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(8, 4), dpi=dpi, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(regions):
    if region == 'root':
        continue
    sns.histplot(data=waveform_df[waveform_df['region'] == region], x='recovery_slope', ax=axs[i],
                 binwidth=2)
    axs[i].set(title=region, xlim=[-200, 0], ylabel='', xlabel='')
f.suptitle('Recovery slope')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'WaveformClassification' / 'rec_slope.jpg',
            dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(8, 4), dpi=dpi, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(regions):
    if region == 'root':
        continue
    sns.histplot(data=waveform_df[waveform_df['region'] == region], x='repolarization_slope', ax=axs[i],
                 binwidth=40)
    axs[i].set(title=region, xlim=[0, 2000], ylabel='', xlabel='')
f.suptitle('Repolarization slope')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['google_drive_fig_path'] / 'WaveformClassification' / 'rep_slope.jpg',
            dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(8, 4), dpi=dpi, sharex=True)
axs = np.concatenate(axs)
for i, region in enumerate(regions):
    if region == 'root':
        continue
    sns.scatterplot(data=waveform_df[waveform_df['region'] == region],
                    x='spike_width', y='peak_trough_ratio',            
                    ax=axs[i])
    axs[i].set(title=region, xlim=[0, 1.5], ylim=[-1, 0], ylabel='', xlabel='')
f.text(0.5, 0.04, 'Spike width (ms)', ha='center')
f.text(0.04, 0.5, 'Peak-to-trough ratio', va='center', rotation='vertical')
sns.despine(trim=True)
plt.subplots_adjust(left=0.1, bottom=0.125, right=0.99, top=0.925)
plt.savefig(path_dict['google_drive_fig_path'] / 'WaveformClassification' / 'width_vs_peaktroughratio.jpg',
            dpi=600)

# %%
f, axs = plt.subplots(2, 4, figsize=(8, 4), dpi=dpi, sharex=True, sharey=True)
axs = np.concatenate(axs)
for i, region in enumerate(regions):
    if region == 'root':
        continue
    sns.scatterplot(data=waveform_df[waveform_df['region'] == region],
                    x='spike_width', y='recovery_slope',            
                    ax=axs[i])
    axs[i].set(title=region, xlim=[0, 1.5], ylim=[-200, 10], ylabel='', xlabel='')
f.text(0.5, 0.04, 'Spike width (ms)', ha='center')
f.text(0.04, 0.5, 'Recovery slope', va='center', rotation='vertical')
sns.despine(trim=True)
plt.subplots_adjust(left=0.1, bottom=0.125, right=0.99, top=0.925)
plt.savefig(path_dict['google_drive_fig_path'] / 'WaveformClassification' / 'width_vs_recslope.jpg',
            dpi=600)