# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:22:58 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import networkx as nx
from msvr_functions import paths, figure_style
colors, dpi = figure_style()

# Load in data
path_dict = paths()
granger_df = pd.read_csv(join(path_dict['save_path'], 'granger_causality_context_50mmbins.csv'))

session_avg = granger_df.groupby(['region1', 'region2', 'object', 'date'])['f_stat'].mean().reset_index()
mean_causality = session_avg.groupby(['region1', 'region2', 'object'])['f_stat'].mean().reset_index()
objects = mean_causality['object'].unique()

# Create an empty list to store results
combined_pval_list = []

# Loop over all unique (region1, region2, object) combinations
grouped = granger_df.groupby(['region1', 'region2', 'object'])

for (region1, region2, obj), group in grouped:
    pvals = group['p_value'].dropna().values
    if len(pvals) > 0:
        _, combined_p = stats.combine_pvalues(pvals, method='fisher')
        #combined_p = stats.binomtest(np.sum(pvals < 0.05), pvals.shape[0], 0.05).pvalue
        combined_pval_list.append({
            'region1': region1,
            'region2': region2,
            'object': obj,
            'combined_p_value': combined_p
        })

# Create DataFrame
mean_causality['p_value'] = pd.DataFrame(combined_pval_list)['combined_p_value']

# Calculate metrics
metrics_df = pd.DataFrame()
for obj in ['object1', 'object2']:
    for region in mean_causality['region1'].unique():

        # Calculate in/out strenght and asymmetry
        this_df = mean_causality[mean_causality['object'] == obj]
        out_strength = np.mean(this_df.loc[this_df['region1'] == region, 'f_stat'])
        in_strength = np.mean(this_df.loc[this_df['region2'] == region, 'f_stat'])
        causal_asymmetry_ratio = (out_strength - in_strength) / (out_strength + in_strength)

        # Add to df
        metrics_df = pd.concat((metrics_df, pd.DataFrame(index=[metrics_df.shape[0]], data={
            'out_strength': out_strength, 'in_strength': in_strength, 'causal_asymmetry_ratio': causal_asymmetry_ratio,
            'region': region, 'object': obj})))

# %%
for obj in ['object1', 'object2']:
    plt.figure(figsize=(2, 2), dpi=300)

    # Build a directed graph
    G = nx.DiGraph()

    # Add weighted edges for significant causality
    for _, row in mean_causality[(mean_causality['object'] == obj) & (mean_causality['p_value'] < 0.05)].iterrows():
        G.add_edge(row['region1'], row['region2'], weight=row['f_stat'])

    # Add all expected nodes explicitly to ensure isolated ones are included
    node_order = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']
    G.add_nodes_from(node_order)

    # Layout
    pos = nx.circular_layout(dict(zip(node_order, range(len(node_order)))))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='lightblue')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=7)

    # Draw edges with weights (causality strength)
    edges = G.edges(data=True)
    weights = [d['weight'] for (_, _, d) in edges]
    scaled_weights = [0.5 + 1 * ((w - min(weights)) / (max(weights) - min(weights))) for w in weights]

    nx.draw_networkx_edges(
        G, pos, edgelist=edges,
        width=scaled_weights,
        min_source_margin=12,
        min_target_margin=12,
        arrows=True,
        arrowsize=10,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.175'  # helps with bidirectionality
    )

    x_values, y_values = zip(*pos.values())
    x_margin = 0.2
    y_margin = 0.2

    plt.xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    plt.ylim(min(y_values) - y_margin, max(y_values) + y_margin)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'granger_causality_{obj}.jpg', dpi=600)
    plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'granger_causality_{obj}.pdf')

# %%

f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)

ax1.plot([1.1, 1.3], [1.1, 1.3], color='grey', ls='--', zorder=0)
for i, region in enumerate(metrics_df['region'].unique()):
    ax1.scatter(metrics_df.loc[(metrics_df['object'] == 'object1') & (metrics_df['region'] == region), 'out_strength'].values[0],
                metrics_df.loc[(metrics_df['object'] == 'object2') & (metrics_df['region'] == region), 'out_strength'].values[0],
                marker='s', color=colors[region], label=region, s=15, zorder=1)
#ax1.legend(bbox_to_anchor=(0.7, 0.65), prop={'size': 6})
ax1.set(xlabel='Out-Strength rew. object 1', ylabel='Out-Strength rew. object 2',
        xticks=[1.1, 1.2, 1.3], yticks=[1.1, 1.2, 1.3])

sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'granger_causality_out_strength.pdf')

# %%
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)

ax1.plot([1.1, 1.3], [1.1, 1.3], color='grey', ls='--', zorder=0)
for i, region in enumerate(metrics_df['region'].unique()):
    ax1.scatter(metrics_df.loc[(metrics_df['object'] == 'object1') & (metrics_df['region'] == region), 'in_strength'].values[0],
                metrics_df.loc[(metrics_df['object'] == 'object2') & (metrics_df['region'] == region), 'in_strength'].values[0],
                marker='s', color=colors[region], label=region, s=15, zorder=1)
#ax1.legend(bbox_to_anchor=(0.7, 0.65), prop={'size': 6})
ax1.set(xlabel='In-Strength rew. object 1', ylabel='In-Strength rew. object 2',
        xticks=[1.1, 1.2, 1.3], yticks=[1.1, 1.2, 1.3])

sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'granger_causality_in_strength.pdf')
