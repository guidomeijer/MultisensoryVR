# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 2026
Author: Guido Meijer

Plots cross-validated R-squared and optimal dimensionality over time for Reduced Rank Regression (RRR)
across directional region pairs for the three objects separately.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from msvr_functions import paths, figure_style

path_dict = paths()
colors, dpi = figure_style()

# Load results
data_path = path_dict['google_drive_data_path'] / 'rrr_results.csv'
rrr_df = pd.read_csv(data_path)

print(f'Loaded {len(rrr_df)} RRR records from {data_path}')

# Get unique region pairs and objects
region_pairs = sorted(rrr_df['region_pair'].unique())
objects = ['obj1', 'obj2', 'obj3']

# Custom color palette for objects
obj_colors = {'obj1': '#1f77b4', 'obj2': '#ff7f0e', 'obj3': '#2ca02c'}

# Filter out pairs with few samples if any
pair_counts = rrr_df.groupby('region_pair').size()
valid_pairs = pair_counts[pair_counts > 10].index
rrr_df = rrr_df[rrr_df['region_pair'].isin(valid_pairs)]

# 1. Plot R^2 over time per object for each directional region pair
n_pairs = len(valid_pairs)
n_cols = 4
n_rows = int(np.ceil(n_pairs / n_cols))

fig_r2, axs_r2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.5), dpi=dpi, sharex=True, sharey=True)
axs_r2 = axs_r2.flatten() if n_pairs > 1 else [axs_r2]

for idx, pair in enumerate(valid_pairs):
    ax = axs_r2[idx]
    pair_df = rrr_df[rrr_df['region_pair'] == pair]
    
    sns.lineplot(
        data=pair_df,
        x='time',
        y='r2',
        hue='object',
        palette=obj_colors,
        ax=ax,
        errorbar='se',
        err_kws={'lw': 0}
    )
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_title(pair, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time from object onset (s)')
    ax.set_ylabel('Cross-validated $R^2$')
    if idx != 0:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

for idx in range(n_pairs, len(axs_r2)):
    fig_r2.delaxes(axs_r2[idx])

sns.despine(trim=True)
plt.tight_layout()

# Save R^2 figure
out_dir = path_dict['paper_fig_path'] / 'InterAreaCommunication'
out_dir.mkdir(parents=True, exist_ok=True)

fig_r2.savefig(out_dir / 'rrr_r2_over_time.pdf', bbox_inches='tight')
fig_r2.savefig(out_dir / 'rrr_r2_over_time.jpg', dpi=600, bbox_inches='tight')
print(f'Saved R^2 plot to {out_dir / "rrr_r2_over_time.pdf"}')

# 2. Plot Dimensionality over time per object for each directional region pair
fig_dim, axs_dim = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.5), dpi=dpi, sharex=True, sharey=True)
axs_dim = axs_dim.flatten() if n_pairs > 1 else [axs_dim]

for idx, pair in enumerate(valid_pairs):
    ax = axs_dim[idx]
    pair_df = rrr_df[rrr_df['region_pair'] == pair]
    
    sns.lineplot(
        data=pair_df,
        x='time',
        y='dimensionality',
        hue='object',
        palette=obj_colors,
        ax=ax,
        errorbar='se',
        err_kws={'lw': 0}
    )
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_title(pair, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time from object onset (s)')
    ax.set_ylabel('Dimensionality ($d_{opt}$)')
    if idx != 0:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

for idx in range(n_pairs, len(axs_dim)):
    fig_dim.delaxes(axs_dim[idx])

sns.despine(trim=True)
plt.tight_layout()

fig_dim.savefig(out_dir / 'rrr_dimensionality_over_time.pdf', bbox_inches='tight')
fig_dim.savefig(out_dir / 'rrr_dimensionality_over_time.jpg', dpi=600, bbox_inches='tight')
print(f'Saved Dimensionality plot to {out_dir / "rrr_dimensionality_over_time.pdf"}')

plt.show()
