# -*- coding: utf-8 -*-
"""
Plot CA1 trajectories and subspace orthogonality per cortical region pair.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from msvr_functions import paths, figure_style

# Initialize
colors, dpi = figure_style()
path_dict = paths()

# Load data
traj_df = pd.read_csv(path_dict['google_drive_data_path'] / 'cortex_to_ca1_trajectories.csv')
orth_df = pd.read_csv(path_dict['google_drive_data_path'] / 'cortex_to_ca1_orthogonality.csv')

# Plot CA1 trajectories
# We will create a plot with components on the rows, and cortical regions on the columns
objects = traj_df['object'].unique()
components = np.sort(traj_df['component'].unique())
regions = traj_df['cortical_region'].unique()

# Check colors dictionary
col_rewarded = colors.get('rewarded', 'green')
col_non_rewarded = colors.get('non-rewarded', 'gray')
palette = {'rewarded': col_rewarded, 'non-rewarded': col_non_rewarded}

for obj in objects:
    f, axs = plt.subplots(len(components), len(regions), figsize=(1.5 * len(regions), 1.5 * len(components)), dpi=dpi, sharex=True, sharey=True)
    
    # Handle single dimension arrays
    if len(components) == 1 and len(regions) == 1:
        axs = np.array([[axs]])
    elif len(components) == 1:
        axs = np.expand_dims(axs, axis=0)
    elif len(regions) == 1:
        axs = np.expand_dims(axs, axis=1)

    for c, comp in enumerate(components):
        for r, region in enumerate(regions):
            ax = axs[c, r]
            
            # Filter data
            subset = traj_df[(traj_df['object'] == obj) & 
                             (traj_df['component'] == comp) & 
                             (traj_df['cortical_region'] == region)]
            
            if len(subset) > 0:
                # Plot rewarded and non-rewarded
                sns.lineplot(data=subset, x='time', y='ca1_projection', hue='reward', errorbar='se', ax=ax,
                             palette=palette)
            
            if c == 0:
                ax.set_title(f'{region}')
            
            if c == len(components) - 1:
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xlabel('')
            
            if r == 0:
                ax.set_ylabel(f'Comp {comp}')
            else:
                ax.set_ylabel('')
                
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    
    sns.despine(trim=True)
    plt.suptitle(f'Object {obj[-1]}', y=1.05)
    plt.tight_layout()
    plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'cortex_to_ca1_trajectories_{obj}.jpg', dpi=600, bbox_inches='tight')
    plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'cortex_to_ca1_trajectories_{obj}.pdf', bbox_inches='tight')
    plt.show()

# Plot dot product (orthogonality angle) per cortical region pair
for obj in objects:
    f, ax = plt.subplots(figsize=(3.5, 2.5), dpi=dpi)
    
    subset = orth_df[orth_df['object'] == obj]
    
    # Calculate mean per region pair and subject
    subj_mean = subset.groupby(['region_pair', 'subject'])['angle_deg'].mean().reset_index()
    
    # Sort pairs alphabetically
    order = sorted(subj_mean['region_pair'].unique())
    
    sns.barplot(data=subj_mean, x='region_pair', y='angle_deg', order=order, errorbar='se', ax=ax, color='lightgray')
    sns.stripplot(data=subj_mean, x='region_pair', y='angle_deg', order=order, ax=ax, color='k', alpha=0.5, size=3, jitter=True)
    
    ax.set_title(f'Object {obj[-1]}')
    ax.set_xlabel('Cortical region pair')
    ax.set_ylabel('Subspace angle (degrees)')
    ax.tick_params(axis='x', rotation=90)
    
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'cortex_to_ca1_orthogonality_{obj}.jpg', dpi=600, bbox_inches='tight')
    plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'cortex_to_ca1_orthogonality_{obj}.pdf', bbox_inches='tight')
    plt.show()


# %% Plot example session trajectories (Component 1) colored by region
# Find session with the most cortical regions represented
session_counts = traj_df.groupby(['subject', 'date'])['cortical_region'].nunique().reset_index()
best_session = session_counts.sort_values('cortical_region', ascending=False).iloc[1]
example_subj = best_session['subject']
example_date = best_session['date']

example_df = traj_df[(traj_df['subject'] == example_subj) & 
                     (traj_df['date'] == example_date) & 
                     (traj_df['component'] == 1)]

for obj in objects:
    obj_df = example_df[example_df['object'] == obj]
    for reward_type in ['rewarded', 'non-rewarded']:
        f, ax = plt.subplots(figsize=(3, 2.5), dpi=dpi)
        subset = obj_df[obj_df['reward'] == reward_type]
        
        # Plot each region
        for region in subset['cortical_region'].unique():
            reg_df = subset[subset['cortical_region'] == region]
            ax.plot(reg_df['time'], reg_df['ca1_projection'], label=region, color=colors.get(region, 'gray'))
            
        ax.set_title(f'Obj {obj[-1]} ({reward_type})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('CA1 Projection (Comp 1)')
        ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'cortex_to_ca1_example_{obj}_{reward_type}.jpg', dpi=600, bbox_inches='tight')
        plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'cortex_to_ca1_example_{obj}_{reward_type}.pdf', bbox_inches='tight')
        plt.show()


