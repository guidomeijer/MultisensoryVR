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
BRIDGE_FAR = 900
FAR_OBJ = 1350
BRIDGE_NEAR = 700
NEAR_OBJ = 900
AFTER_OBJ = 70

# Initialize
path_dict = paths(sync=False)
subjects = load_subjects()

# Create hypothesis RDMs
object_rdm = np.array([[9, 0, 1, 1],
                       [0, 9, 1, 1],
                       [1, 1, 9, 0],
                       [1, 1, 0, 9]])
reward_rdm = np.array([[9, 2, 2, 0],
                       [2, 9, 0, 2],
                       [2, 0, 9, 2],
                       [0, 2, 2, 9]])
state_rdm = np.array([[9, 2, 0, 2],
                      [2, 9, 2, 0],
                      [0, 2, 9, 2],
                      [2, 0, 2, 9]])

# Load in processed data
with open(path_dict['google_drive_data_path'] / 'residuals_position_0mms.pickle', 'rb') as handle:
    spike_dict = pickle.load(handle)

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
            pop_vec['A   A'] = np.mean(spike_counts[(spatial_bins == FIRST_OBJ + AFTER_OBJ) & (context_per_bin == obj1_goal), :], axis=0)
            pop_vec['A   B'] = np.mean(spike_counts[(spatial_bins == FIRST_OBJ + AFTER_OBJ) & (context_per_bin == 3 - obj1_goal), :], axis=0)
            if is_far:
                pop_vec['B   A'] = np.mean(spike_counts[(spatial_bins == FAR_OBJ + AFTER_OBJ) & (context_per_bin == obj1_goal), :], axis=0)
                pop_vec['B   B'] = np.mean(spike_counts[(spatial_bins == FAR_OBJ + AFTER_OBJ) & (context_per_bin == 3 - obj1_goal), :], axis=0)
            else:
                pop_vec['B   A'] = np.mean(spike_counts[(spatial_bins == NEAR_OBJ + AFTER_OBJ) & (context_per_bin == obj1_goal), :], axis=0)
                pop_vec['B   B'] = np.mean(spike_counts[(spatial_bins == NEAR_OBJ + AFTER_OBJ) & (context_per_bin == 3 - obj1_goal), :], axis=0)

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
            r_state, p_state = stats.spearmanr(rdm[np.triu_indices(rdm.shape[0], k=1)], state_rdm[np.triu_indices(rdm.shape[0], k=1)])

            # Multiple Regression
            X = np.column_stack((
                object_rdm[np.triu_indices(rdm.shape[0], k=1)],
                reward_rdm[np.triu_indices(rdm.shape[0], k=1)],
                state_rdm[np.triu_indices(rdm.shape[0], k=1)]
            ))
            y = rdm[np.triu_indices(rdm.shape[0], k=1)]
            
            # Standardize X and y to get standardized beta coefficients
            X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            y_std = (y - np.mean(y)) / np.std(y)
            
            # Interaction terms
            int_obj_rew = X_std[:, 0] * X_std[:, 1]
            int_obj_state = X_std[:, 0] * X_std[:, 2]
            int_rew_state = X_std[:, 1] * X_std[:, 2]
            
            # Add interaction terms to X_std
            X_std_full = np.column_stack((X_std, int_obj_rew, int_obj_state, int_rew_state))
            
            # Perform multiple linear regression
            reg = LinearRegression().fit(X_std_full, y_std)
            beta_obj, beta_rew, beta_state, beta_obj_rew, beta_obj_state, beta_rew_state = reg.coef_

            # Add to dicts and dfs
            corr_df = pd.concat((corr_df, pd.DataFrame(data={
                'p_object': [p_obj], 'r_object': [r_obj], 'r_reward': [r_rew], 'p_reward': [p_rew],
                'p_state': [p_state], 'r_state': [r_state],
                'beta_object': [beta_obj], 'beta_reward': [beta_rew], 'beta_state': [beta_state],
                'beta_obj_rew': [beta_obj_rew], 'beta_obj_state': [beta_obj_state], 'beta_rew_state': [beta_rew_state],
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
plot_min = 0.7
plot_max = 1.2

f, axs = plt.subplots(1, 3, figsize=(3.3, 1.6), dpi=dpi, sharey=True)

# Create a copy of the colormap and set NaN values to white
cmap_hypo = plt.get_cmap(use_cmap).copy()
cmap_hypo.set_bad('white')

# Plot hypothesis RDMs with diagonal set to NaN for white display
object_rdm_plot = object_rdm.astype(float)
reward_rdm_plot = reward_rdm.astype(float)
state_rdm_plot = state_rdm.astype(float)
np.fill_diagonal(object_rdm_plot, np.nan)
np.fill_diagonal(reward_rdm_plot, np.nan)
np.fill_diagonal(state_rdm_plot, np.nan)

axs[0].imshow(object_rdm_plot, cmap=cmap_hypo, clim=[0, 2])
axs[0].set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, title='Object')
axs[0].tick_params('x', rotation=90)
axs[0].text(-3.7, -1, 'Object', ha='center', va='center', clip_on=False, weight='bold')
axs[0].text(-1, -1, 'State', ha='center', va='center', clip_on=False, weight='bold')
axs[1].imshow(reward_rdm_plot, cmap=cmap_hypo, clim=[0, 2])
axs[1].set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, title='Reward')
axs[1].tick_params('x', rotation=90)
axs[2].imshow(state_rdm_plot, cmap=cmap_hypo, clim=[0, 2])
axs[2].set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, title='State')
axs[2].tick_params('x', rotation=90)
plt.tight_layout()
#plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'hypotheses.jpg', dpi=600)
#plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'hypotheses.pdf')
plt.show()

# %%
f, axs = plt.subplots(1, 6, figsize=(7, 1.75), dpi=dpi, sharey=True)

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
axs[0].text(-4, -1, 'Object', ha='center', va='center', clip_on=False, weight='bold')
axs[0].text(-1.5, -1, 'State', ha='center', va='center', clip_on=False, weight='bold')

sm = plt.cm.ScalarMappable(cmap=use_cmap, norm=plt.Normalize(vmin=plot_min, vmax=plot_max))
sm.set_array([])
cbar = f.colorbar(sm, ax=axs, orientation='vertical', fraction=0.012)
cbar.set_label('Dissimilarity (1-r)', rotation=270, labelpad=10)

plt.subplots_adjust(left=0.12, bottom=0.25, right=0.85, top=1)
#plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_dissimilatrity.jpg', dpi=600)
#plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_dissimilatrity.pdf')
plt.show()

# %% Plot Multiple Regression Beta Weights
f, axs = plt.subplots(1, 3, figsize=(1.2*3, 1.75), dpi=dpi, sharey=False)

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
    axs[0].text(i, 0.6, label, ha='center', va='center', fontsize=8)
axs[0].set(title='Object', ylabel='Beta weight', yticks=[-0.4, -0.2, 0, 0.2, 0.4, 0.6],
            yticklabels=[-0.4, -0.2, 0, 0.2, 0.4, 0.6], ylim=[-0.4, 0.6], xlabel='')  
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

region_order = corr_df.groupby('region').mean(numeric_only=True)['beta_state'].sort_values(ascending=False).index.values
p_values = np.full(len(region_order), np.nan)
for i, region in enumerate(region_order):
    p_values[i] = stats.ttest_1samp(corr_df.loc[corr_df['region'] == region, 'beta_state'].values, 0)[1]  
sns.barplot(data=corr_df, x='region', y='beta_state', ax=axs[2], hue='region', palette=colors, errorbar='se',
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
axs[2].set(title='State', ylabel='', xlabel='', yticks=[-0.3, 0, 0.3],
           yticklabels=[-0.3, 0, 0.3], ylim=[-0.3, 0.3])
axs[2].tick_params('x', rotation=90)

sns.despine(trim=True)
plt.tight_layout()
#plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_regression.jpg', dpi=600)
#plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_regression.pdf')
plt.show()

# %% Plot Multiple Regression Interaction Beta Weights
f, axs = plt.subplots(1, 3, figsize=(1.2*3, 1.75), dpi=dpi, sharey=False)

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
    axs[0].text(i, 0.2, label, ha='center', va='center', fontsize=8)
axs[0].set(title='Object x Reward', ylabel='Interaction Beta Weight', xlabel='', yticks=[-0.4, -0.2, 0, 0.2],
           yticklabels=[-0.4, -0.2, 0, 0.2], ylim=[-0.4, 0.2])
axs[0].tick_params('x', rotation=90)

region_order = corr_df.groupby('region').mean(numeric_only=True)['beta_obj_state'].sort_values(ascending=False).index.values
p_values = np.full(len(region_order), np.nan)
for i, region in enumerate(region_order):
    p_values[i] = stats.ttest_1samp(corr_df.loc[corr_df['region'] == region, 'beta_obj_state'].values, 0)[1]  
sns.barplot(data=corr_df, x='region', y='beta_obj_state', ax=axs[1], hue='region', palette=colors, errorbar='se',
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
    axs[1].text(i, 0.28, label, ha='center', va='center', fontsize=8)
axs[1].set(title='Object x State', ylabel='', xlabel='', yticks=[-0.3, 0, 0.3],
           yticklabels=[-0.3, 0, 0.3], ylim=[-0.3, 0.3])
axs[1].tick_params('x', rotation=90)

region_order = corr_df.groupby('region').mean(numeric_only=True)['beta_rew_state'].sort_values(ascending=False).index.values
p_values = np.full(len(region_order), np.nan)
for i, region in enumerate(region_order):
    p_values[i] = stats.ttest_1samp(corr_df.loc[corr_df['region'] == region, 'beta_rew_state'].values, 0)[1]  
sns.barplot(data=corr_df, x='region', y='beta_rew_state', ax=axs[2], hue='region', palette=colors, errorbar='se',
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
    axs[2].text(i, -0.58, label, ha='center', va='center', fontsize=8)
axs[2].set(title='Reward x State', ylabel='', xlabel='', yticks=[-0.6, -0.4, -0.2, 0],
           yticklabels=[-0.6, -0.4, -0.2, 0], ylim=[-0.6, 0])
axs[2].tick_params('x', rotation=90)

sns.despine(trim=True)
plt.tight_layout()
#plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_interaction.jpg', dpi=600)
#plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_interaction.pdf')
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
#plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_similarity_analysis.jpg', dpi=600)
#plt.savefig(path_dict['paper_fig_path'] / 'Representations' / 'representation_similarity_analysis.pdf')
plt.show()
