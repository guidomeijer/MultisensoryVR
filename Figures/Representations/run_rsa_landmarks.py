# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 09/06/2026
"""
# %%

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import networkx as nx
from msvr_functions import paths, load_subjects, figure_style, load_objects
colors, dpi = figure_style()

# Settings
MIN_NEURONS = 10
FIRST_OBJ = 450
CONTROL_FAR = 900
FAR_OBJ = 1350
CONTROL_NEAR = 1350
NEAR_OBJ = 900
AFTER_OBJ = 70

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()

# Create hypothesis RDMs
object_rdm = np.array([[9, 0, 1, 1, 1, 1],
                       [0, 9, 1, 1, 1, 1],
                       [1, 1, 9, 0, 1, 1],
                       [1, 1, 0, 9, 1, 1],
                       [1, 1, 1, 1, 9, 0],
                       [1, 1, 1, 1, 0, 9]])
reward_rdm = np.array([[9, 2, 2, 0, 1, 1],
                       [2, 9, 0, 2, 1, 1],
                       [2, 0, 9, 2, 1, 1],
                       [0, 2, 2, 9, 1, 1],
                       [1, 1, 1, 1, 9, 1],
                       [1, 1, 1, 1, 1, 9]])
context_rdm = np.array([[9, 2, 0, 2, 0, 2],
                      [2, 9, 2, 0, 2, 0],
                      [0, 2, 9, 2, 0, 2],
                      [2, 0, 2, 9, 2, 0],
                      [0, 2, 0, 2, 9, 2],
                      [2, 0, 2, 0, 2, 9]])

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position_0mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)
    
#with open(path_dict['google_drive_data_path'] / 'residuals_motor.pickle', 'rb') as handle:
#    spike_dict = pickle.load(handle)

# Load recording info
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv')
rec['date'] = rec['date'].astype(str)

# Loop over recordings
rsa_df, corr_df = pd.DataFrame(), pd.DataFrame()
all_rdms = {'VIS': [], 'AUD': [], 'TEa': [], 'PERI': [], 'LEC': [], 'CA1': []}
for i, this_ses in enumerate(np.unique(spike_dict['date'])):

    # Loop over probes
    rdm_regions = dict()
    for j in np.where(np.array(spike_dict['date']) == this_ses)[0]:

        # Get subject
        subject = spike_dict['subject'][j]
        is_far = subjects.loc[subjects['SubjectID'] == subject, 'Far'].values[0]
        obj_df = load_objects(subject, this_ses)

        # Get which context is the rewarded context for the first and second object
        obj1_goal = obj_df.loc[(obj_df['object'] == 1) & (obj_df['goal'] == 1), 'sound'].values[0]
        obj2_goal = obj_df.loc[(obj_df['object'] == 2) & (obj_df['goal'] == 1), 'sound'].values[0]

        # Get brain regions
        regions = spike_dict['region'][j]
        if regions is None:
            continue
        unique_regions = np.unique(regions)

        # Loop over brain regions
        for r, region in enumerate(unique_regions):
            if region == 'root':
                continue

            # Select neurons from this brain region
            region_mask = regions == region
            spike_counts = spike_dict['residuals'][j][:, region_mask]  # spatial bins x neurons
            spike_counts = spike_counts[:, np.std(spike_counts, axis=0) > 0.5] # Throw out silent neurons
            if spike_counts.shape[1] < MIN_NEURONS:
                continue
            spike_counts = (spike_counts - np.mean(spike_counts, axis=0)) / np.std(spike_counts, axis=0) # zscore
            spatial_bins = spike_dict['position'][j]
            context_per_bin = spike_dict['context'][j]


            # Get average population vector per condition
            pop_vec = dict()
            first_obj_mask = spatial_bins == spatial_bins[np.argmin(np.abs(spatial_bins - (FIRST_OBJ + AFTER_OBJ)))]
            near_obj_mask = spatial_bins == spatial_bins[np.argmin(np.abs(spatial_bins - (NEAR_OBJ + AFTER_OBJ)))]
            far_obj_mask = spatial_bins == spatial_bins[np.argmin(np.abs(spatial_bins - (FAR_OBJ + AFTER_OBJ)))]
            control_far_mask = spatial_bins == spatial_bins[np.argmin(np.abs(spatial_bins - (CONTROL_FAR + AFTER_OBJ)))]
            control_near_mask = spatial_bins == spatial_bins[np.argmin(np.abs(spatial_bins - (CONTROL_NEAR + AFTER_OBJ)))]
            pop_vec['A   A'] = np.mean(spike_counts[first_obj_mask & (context_per_bin == obj1_goal), :], axis=0)
            pop_vec['A   B'] = np.mean(spike_counts[first_obj_mask & (context_per_bin == 3 - obj1_goal), :], axis=0)
            if is_far:
                pop_vec['B   A'] = np.mean(spike_counts[far_obj_mask & (context_per_bin == obj1_goal), :], axis=0)
                pop_vec['B   B'] = np.mean(spike_counts[far_obj_mask & (context_per_bin == 3 - obj1_goal), :], axis=0)
                pop_vec['C   A'] = np.mean(spike_counts[control_far_mask & (context_per_bin == obj1_goal), :], axis=0)
                pop_vec['C   B'] = np.mean(spike_counts[control_far_mask & (context_per_bin == 3 - obj1_goal), :], axis=0)
            else:
                pop_vec['B   A'] = np.mean(spike_counts[near_obj_mask & (context_per_bin == obj1_goal), :], axis=0)
                pop_vec['B   B'] = np.mean(spike_counts[near_obj_mask & (context_per_bin == 3 - obj1_goal), :], axis=0)
                pop_vec['C   A'] = np.mean(spike_counts[control_near_mask & (context_per_bin == obj1_goal), :], axis=0)
                pop_vec['C   B'] = np.mean(spike_counts[control_near_mask & (context_per_bin == 3 - obj1_goal), :], axis=0)

            # Construct representation dissimilarity matrix (RDM)
            labels = list(pop_vec.keys())
            rdm = np.ones((len(labels), len(labels)))*2
            for i_idx, label_i in enumerate(labels):
                for j_idx, label_j in enumerate(labels):
                    if i_idx == j_idx:
                        continue
                    if np.all(np.isnan(pop_vec[label_i])) or np.all(np.isnan(pop_vec[label_j])):
                        rdm[i_idx, j_idx] = np.nan
                    else:
                        rdm[i_idx, j_idx] = 1 - np.corrcoef(pop_vec[label_i], pop_vec[label_j])[0, 1]

            # Correlate with hypothesis representations
            r_obj, p_obj = stats.spearmanr(rdm[np.triu_indices(rdm.shape[0], k=1)], object_rdm[np.triu_indices(rdm.shape[0], k=1)])
            r_rew, p_rew = stats.spearmanr(rdm[np.triu_indices(rdm.shape[0], k=1)], reward_rdm[np.triu_indices(rdm.shape[0], k=1)])
            r_context, p_context = stats.spearmanr(rdm[np.triu_indices(rdm.shape[0], k=1)], context_rdm[np.triu_indices(rdm.shape[0], k=1)])

            # Multiple Regression
            X = np.column_stack((
                object_rdm[np.triu_indices(rdm.shape[0], k=1)],
                reward_rdm[np.triu_indices(rdm.shape[0], k=1)],
                context_rdm[np.triu_indices(rdm.shape[0], k=1)]
            ))
            y = rdm[np.triu_indices(rdm.shape[0], k=1)]
            
            # Standardize X and y to get standardized beta coefficients
            X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            y_std = (y - np.mean(y)) / np.std(y)
            
            # Interaction terms
            int_obj_rew = X_std[:, 0] * X_std[:, 1]
            int_obj_context = X_std[:, 0] * X_std[:, 2]
            int_rew_context = X_std[:, 1] * X_std[:, 2]
            
            # Add interaction terms to X_std
            X_std_full = np.column_stack((X_std, int_obj_rew, int_obj_context, int_rew_context))
            
            # Perform multiple linear regression
            reg = LinearRegression().fit(X_std_full, y_std)
            beta_obj, beta_rew, beta_context, beta_obj_rew, beta_obj_context, beta_rew_context = reg.coef_

            # Add to dicts and dfs
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'p_object': [p_obj], 'r_object': [r_obj], 'r_reward': [r_rew], 'p_reward': [p_rew],
                'p_context': [p_context], 'r_context': [r_context],
                'beta_object': [beta_obj], 'beta_reward': [beta_rew], 'beta_context': [beta_context],
                'beta_obj_rew': [beta_obj_rew], 'beta_obj_context': [beta_obj_context], 'beta_rew_context': [beta_rew_context],
                'region': [region], 'subject': [subject], 'date': [this_ses]})))
            rdm_regions[region] = rdm
            all_rdms[region].append(rdm)

    # Sort pairs alphabetically so that there are never any duplicates
    region_pairs = sorted([tuple(sorted(pair)) for pair in list(combinations(rdm_regions.keys(), 2))])

    # RSA analysis: correlate RDMs of all region pairs
    for pair in region_pairs:

        # Flatten the upper triangle of the RDMs to get the distance vectors
        rdm1 = rdm_regions[pair[0]][np.triu_indices(len(labels), k=1)]
        rdm2 = rdm_regions[pair[1]][np.triu_indices(len(labels), k=1)]

        # Calculate correlation between RDMs
        if np.any(np.isnan(rdm1)) or np.any(np.isnan(rdm2)):
            rsa_corr = np.nan
        else:
            rsa_corr = np.corrcoef(rdm1, rdm2)[0, 1]

        # Store results
        rsa_df = pd.concat([rsa_df, pd.DataFrame({
            'subject': [subject], 'date': [this_ses],
            'region1': [pair[0]], 'region2': [pair[1]],
            'rsa_correlation': [rsa_corr]
        })], ignore_index=True)


# %% Plot results
use_cmap = 'viridis_r'

f, axs = plt.subplots(1, 3, figsize=(3.3, 1.6), dpi=dpi, sharey=True)

# Create a copy of the colormap and set NaN values to white
cmap_hypo = plt.get_cmap(use_cmap).copy()
cmap_hypo.set_bad('white')

# Plot hypothesis RDMs with diagonal set to NaN for white display
object_rdm_plot = object_rdm.astype(float)
reward_rdm_plot = reward_rdm.astype(float)
context_rdm_plot = context_rdm.astype(float)
np.fill_diagonal(object_rdm_plot, np.nan)
np.fill_diagonal(reward_rdm_plot, np.nan)
np.fill_diagonal(context_rdm_plot, np.nan)

axs[0].imshow(object_rdm_plot, cmap=cmap_hypo, clim=[0, 2])
axs[0].set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, title='Object')
axs[0].tick_params('x', rotation=90)
axs[0].text(-3.5, -1, 'Object', ha='center', va='center', clip_on=False, weight='bold')
axs[0].text(-1, -1, 'Context', ha='center', va='center', clip_on=False, weight='bold')
axs[1].imshow(reward_rdm_plot, cmap=cmap_hypo, clim=[0, 2])
axs[1].set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, title='Reward')
axs[1].tick_params('x', rotation=90)
axs[2].imshow(context_rdm_plot, cmap=cmap_hypo, clim=[0, 2])
axs[2].set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, title='Context')
axs[2].tick_params('x', rotation=90)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'hypotheses.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'hypotheses.pdf')
plt.show()

# %%
plot_min = 0.7
plot_max = 1.3

f, axs = plt.subplots(1, 6, figsize=(6.5, 1.5), dpi=dpi, sharey=True)

# Create a copy of the colormap and set NaN values to white for the mean RDMs
cmap_rdm = plt.get_cmap(use_cmap).copy()
cmap_rdm.set_bad('white')

for i, plot_region in enumerate(all_rdms.keys()):
    mean_rdm = np.nanmean(np.dstack(all_rdms[plot_region]), axis=2)
    np.fill_diagonal(mean_rdm, np.nan) # Set diagonal to NaN for white display
    axs[i].imshow(mean_rdm, vmin=plot_min, vmax=plot_max, cmap=cmap_rdm)
    #axs[i].invert_yaxis()
    axs[i].set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
               xticklabels=labels, yticklabels=labels, title=plot_region)
    axs[i].tick_params('x', rotation=90)

sm = plt.cm.ScalarMappable(cmap=use_cmap, norm=plt.Normalize(vmin=plot_min, vmax=plot_max))
sm.set_array([])
cbar_ax = f.add_axes([0.87, 0.4, 0.008, 0.45])
cbar = f.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('Dissimilarity (1-r)', rotation=270, labelpad=10)

plt.subplots_adjust(left=0.05, bottom=0.25, right=0.85, top=1)
plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_dissimilatrity.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_dissimilatrity.pdf')
plt.show()

# %% Plot Multiple Regression Beta Weights
f, axs = plt.subplots(1, 3, figsize=(3.4, 1.75), dpi=dpi, sharey=False)

region_order = corr_df.groupby('region').mean(numeric_only=True)['beta_object'].sort_values(ascending=False).index.values
p_values = np.full(len(region_order), np.nan)
for i, region in enumerate(region_order):
    p_values[i] = stats.ttest_1samp(corr_df.loc[corr_df['region'] == region, 'beta_object'].values, 0)[1]  
ax = sns.barplot(data=corr_df, x='region', y='beta_object', ax=axs[0], hue='region', palette=colors, errorbar='se',
            order=region_order)
for i, p in enumerate(p_values):
    if p < 0.001:
        label = '***'
    elif p < 0.01:
        label = '**'
    elif p < 0.05:
        label = '*'
    else:
        label = ''
    axs[0].text(i, 0.4, label, ha='center', va='center', fontsize=8)
axs[0].set(title='Object', ylabel='Beta weight', yticks=[-0.4, -0.2, 0, 0.2, 0.4],
            yticklabels=[-0.4, -0.2, 0, 0.2, 0.4], ylim=[-0.4, 0.4], xlabel='')  
axs[0].tick_params('x', rotation=90)

region_order = corr_df.groupby('region').mean(numeric_only=True)['beta_reward'].sort_values(ascending=False).index.values
p_values = np.full(len(region_order), np.nan)
for i, region in enumerate(region_order):
    p_values[i] = stats.ttest_1samp(corr_df.loc[corr_df['region'] == region, 'beta_reward'].values, 0)[1]  
sns.barplot(data=corr_df, x='region', y='beta_reward', ax=axs[1], hue='region', palette=colors, errorbar='se',
            order=region_order)
for i, p in enumerate(p_values):
    if p < 0.001:
        label = '***'
    elif p < 0.01:
        label = '**'
    elif p < 0.05:
        label = '*'
    else:
        label = ''
    axs[1].text(i, 0.8, label, ha='center', va='center', fontsize=8)
axs[1].set(title='Reward', ylabel='', xlabel='', yticks=[-0.2, 0, 0.2, 0.4, 0.6, 0.8],
           yticklabels=[-0.2, 0, 0.2, 0.4, 0.6, 0.8], ylim=[-0.2, 0.8])
axs[1].tick_params('x', rotation=90)

region_order = corr_df.groupby('region').mean(numeric_only=True)['beta_context'].sort_values(ascending=False).index.values
p_values = np.full(len(region_order), np.nan)
for i, region in enumerate(region_order):
    p_values[i] = stats.ttest_1samp(corr_df.loc[corr_df['region'] == region, 'beta_context'].values, 0)[1]  
sns.barplot(data=corr_df, x='region', y='beta_context', ax=axs[2], hue='region', palette=colors, errorbar='se',
            order=region_order)
for i, p in enumerate(p_values):
    if p < 0.001:
        label = '***'
    elif p < 0.01:
        label = '**'
    elif p < 0.05:
        label = '*'
    else:
        label = ''
    axs[2].text(i, 0.3, label, ha='center', va='center', fontsize=8)
axs[2].set(title='Context', ylabel='', xlabel='', yticks=[-0.1, 0, 0.1, 0.2, 0.3],
           yticklabels=[-0.1, 0, 0.1, 0.2, 0.3], ylim=[-0.1, 0.3])
axs[2].tick_params('x', rotation=90)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_regression.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_regression.pdf')
plt.show()

# %% Plot Multiple Regression Interaction Beta Weights
f, axs = plt.subplots(1, 3, figsize=(3.2, 1.75), dpi=dpi, sharey=True)

region_order = corr_df.groupby('region').mean(numeric_only=True)['beta_obj_rew'].sort_values(ascending=False).index.values
p_values = np.full(len(region_order), np.nan)
for i, region in enumerate(region_order):
    p_values[i] = stats.ttest_1samp(corr_df.loc[corr_df['region'] == region, 'beta_obj_rew'].values, 0)[1]  
sns.barplot(data=corr_df, x='region', y='beta_obj_rew', ax=axs[0], hue='region', palette=colors, errorbar='se',
            order=region_order)
for i, p in enumerate(p_values):
    if p < 0.001:
        label = '***'
    elif p < 0.01:
        label = '**'
    elif p < 0.05:
        label = '*'
    else:
        label = ''
    axs[0].text(i, 0.48, label, ha='center', va='center', fontsize=8)
axs[0].set(title='Object x Reward', ylabel='Interaction Beta Weight', xlabel='', yticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
           yticklabels=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5], ylim=[-0.3, 0.5])
axs[0].tick_params('x', rotation=90)

region_order = corr_df.groupby('region').mean(numeric_only=True)['beta_obj_context'].sort_values(ascending=False).index.values
p_values = np.full(len(region_order), np.nan)
for i, region in enumerate(region_order):
    p_values[i] = stats.ttest_1samp(corr_df.loc[corr_df['region'] == region, 'beta_obj_context'].values, 0)[1]  
sns.barplot(data=corr_df, x='region', y='beta_obj_context', ax=axs[1], hue='region', palette=colors, errorbar='se',
            order=region_order)
for i, p in enumerate(p_values):
    if p < 0.001:
        label = '***'
    elif p < 0.01:
        label = '**'
    elif p < 0.05:
        label = '*'
    else:
        label = ''
    axs[1].text(i, 0.48, label, ha='center', va='center', fontsize=8)
axs[1].set(title='Object x context', ylabel='', xlabel='')
axs[1].tick_params('x', rotation=90)

region_order = corr_df.groupby('region').mean(numeric_only=True)['beta_rew_context'].sort_values(ascending=False).index.values
p_values = np.full(len(region_order), np.nan)
for i, region in enumerate(region_order):
    p_values[i] = stats.ttest_1samp(corr_df.loc[corr_df['region'] == region, 'beta_rew_context'].values, 0)[1]  
sns.barplot(data=corr_df, x='region', y='beta_rew_context', ax=axs[2], hue='region', palette=colors, errorbar='se',
            order=region_order)
for i, p in enumerate(p_values):
    if p < 0.001:
        label = '***'
    elif p < 0.01:
        label = '**'
    elif p < 0.05:
        label = '*'
    else:
        label = ''
    axs[2].text(i, 0.48, label, ha='center', va='center', fontsize=8)
axs[2].set(title='Reward x Context', ylabel='', xlabel='')
axs[2].tick_params('x', rotation=90)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_interaction.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_interaction.pdf')
plt.show()

#%%
# Calculate mean RSA correlation between region pairs
mean_rsa = rsa_df.groupby(['region1', 'region2'])['rsa_correlation'].mean().reset_index()

# Initialize graph
G = nx.Graph()

# Add edges with weights
for _, row in mean_rsa.iterrows():
    if not np.isnan(row['rsa_correlation']):
        G.add_edge(row['region1'], row['region2'], weight=row['rsa_correlation'])

# Set up plot
plt.figure(figsize=(2.2, 1.75), dpi=350)
node_order = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']
pos = nx.circular_layout(dict(zip(node_order, range(len(node_order)))))

# Sort edges by weight to ensure high correlation edges are drawn on top
sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
edge_list = [(u, v) for u, v, d in sorted_edges]
weights_sorted = [d['weight'] for u, v, d in sorted_edges]

# Draw the network
nodes = nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
edges = nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=[w * 5 for w in weights_sorted],
                               edge_color=weights_sorted, edge_cmap=plt.cm.Reds)
nx.draw_networkx_labels(G, pos, font_size=7)

# Add colorbar and title
sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0.2, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_label('Representation similarity', rotation=270, labelpad=10)
plt.axis('off')
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_similarity_analysis.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_similarity_analysis.pdf')
plt.show()


# %% Plot Grouped Bar Chart for Multiple Regression Beta Weights

# 1. Define conditions and layout parameters
conditions = ['beta_object', 'beta_reward', 'beta_context']
cond_labels = ['Object', 'Reward', 'Context']
condition_colors = ['#7bc0a3', '#e8a87c', '#92a8d1']  # Unique color per condition

# 2. Calculate summary statistics
means = corr_df.groupby('region')[conditions].mean()
sems = corr_df.groupby('region')[conditions].sem()

# Establish a single fixed order for the X-axis (sorting by overall average effect)
region_order = ['AUD', 'VIS', 'TEa', 'PERI', 'LEC', 'CA1']

# 3. Set up cluster geometry
x = np.arange(len(region_order))  # Total number of region clusters
width = 0.25                      # Width of individual bars
offsets = [-width, 0, width]      # X-axis shift for each condition within a cluster

# 4. Initialize Plot
fig, ax = plt.subplots(figsize=(3.5, 1.5), dpi=600)
#ax.axhline(0, color='gray', linestyle='-', linewidth=0.6, zorder=1)

# 5. Plot groups and calculate significance
for j, (cond, label) in enumerate(zip(conditions, cond_labels)):
    y_means = [means.loc[reg, cond] for reg in region_order]
    y_sems = [sems.loc[reg, cond] for reg in region_order]
    x_pos = x + offsets[j]
    
    # Plot the bars for the current condition across all regions
    ax.bar(x_pos, y_means, width, yerr=y_sems, label=label, 
           color=condition_colors[j], zorder=2,
           error_kw=dict(ecolor='dimgray', elinewidth=1.1, capsize=0))
    
    # Calculate and add significance stars dynamically above each bar
    for i, reg in enumerate(region_order):
        vals = corr_df.loc[corr_df['region'] == reg, cond].dropna().values
        _, p = stats.ttest_1samp(vals, 0) if len(vals) > 1 else (None, np.nan)
        
        if p < 0.001:
            star = '***'
        elif p < 0.01:
            star = '**'
        elif p < 0.05:
            star = '*'
        else:
            star = ''
            
        if star:
            # Place the star just above the positive bar top or error bar ceiling
            y_star = max(0, y_means[i] + y_sems[i]) + 0.02 if y_means[i] >= 0 else 0.02
            ax.text(x_pos[i], y_star, star, ha='center', va='bottom', fontsize=7, color='black')

# 6. Styling and Aesthetics
ax.set_xticks(x)
ax.set_xticklabels(region_order, rotation=90)
ax.set_ylabel('Beta weight')
ax.set_xlabel('')

# Dynamically pad ylim to accommodate the highest star
ax.set_ylim([-0.3, 1.0]) 
ax.set_yticks([-0.3, 0, 0.3, 0.6, 0.9])
ax.set_yticklabels([-0.3, 0, 0.3, 0.6, 0.9])

# Clean legend positioned out of the way of the data data
#ax.legend(frameon=False, bbox_to_anchor=(0.5, 0.7))

sns.despine(trim=True)
plt.tight_layout()

# 7. Save figures
output_dir = path_dict['paper_fig_path'] / 'Representations'
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'representation_regression_grouped.jpg', dpi=600)
plt.savefig(output_dir / 'representation_regression_grouped.pdf')
plt.show()

# %% Plot Grouped Bar Chart for Interactions

# 1. Define conditions and layout parameters
conditions = ['beta_obj_rew', 'beta_obj_context', 'beta_rew_context']
cond_labels = ['Object', 'Reward', 'Context']
condition_colors = ['#7bc0a3', '#e8a87c', '#92a8d1']  # Unique color per condition

# 2. Calculate summary statistics
means = corr_df.groupby('region')[conditions].mean()
sems = corr_df.groupby('region')[conditions].sem()

# Establish a single fixed order for the X-axis (sorting by overall average effect)
region_order = ['AUD', 'VIS', 'TEa', 'PERI', 'LEC', 'CA1']

# 3. Set up cluster geometry
x = np.arange(len(region_order))  # Total number of region clusters
width = 0.25                      # Width of individual bars
offsets = [-width, 0, width]      # X-axis shift for each condition within a cluster

# 4. Initialize Plot
fig, ax = plt.subplots(figsize=(3.5, 1.5), dpi=600)
#ax.axhline(0, color='gray', linestyle='-', linewidth=0.6, zorder=1)

# 5. Plot groups and calculate significance
for j, (cond, label) in enumerate(zip(conditions, cond_labels)):
    y_means = [means.loc[reg, cond] for reg in region_order]
    y_sems = [sems.loc[reg, cond] for reg in region_order]
    x_pos = x + offsets[j]
    
    # Plot the bars for the current condition across all regions
    ax.bar(x_pos, y_means, width, yerr=y_sems, label=label, 
           color=condition_colors[j], zorder=2,
           error_kw=dict(ecolor='dimgray', elinewidth=1.1, capsize=0))
    
    # Calculate and add significance stars dynamically above each bar
    for i, reg in enumerate(region_order):
        vals = corr_df.loc[corr_df['region'] == reg, cond].dropna().values
        _, p = stats.ttest_1samp(vals, 0) if len(vals) > 1 else (None, np.nan)
        
        if p < 0.001:
            star = '***'
        elif p < 0.01:
            star = '**'
        elif p < 0.05:
            star = '*'
        else:
            star = ''
            
        if star:
            # Place the star just above the positive bar top or error bar ceiling
            y_star = max(0, y_means[i] + y_sems[i]) if y_means[i] >= 0 else 0
            ax.text(x_pos[i], y_star, star, ha='center', va='bottom', fontsize=7, color='black')

# 6. Styling and Aesthetics
ax.set_xticks(x)
ax.set_xticklabels(region_order, rotation=90)
ax.set_ylabel('Interaction beta weight')
ax.set_xlabel('')

# Dynamically pad ylim to accommodate the highest star
ax.set_ylim([-0.3, 0.5]) 
ax.set_yticks([-0.3, 0, 0.3, 0.6])
ax.set_yticklabels([-0.3, 0, 0.3, 0.6])

# Clean legend positioned out of the way of the data data
#ax.legend(frameon=False, bbox_to_anchor=(0.5, 0.7))

sns.despine(trim=True)
plt.tight_layout()

# 7. Save figures
output_dir = path_dict['paper_fig_path'] / 'Representations'
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'representation_regression_interaction.jpg', dpi=600)
plt.savefig(output_dir / 'representation_regression_interaction.pdf')
plt.show()
