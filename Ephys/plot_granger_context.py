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
from mne_connectivity.viz import plot_connectivity_circle
from msvr_functions import paths, load_subjects, figure_style, combine_regions
colors, dpi = figure_style()

# Load in data
path_dict = paths()
granger_df = pd.read_csv(join(path_dict['save_path'], 'granger_causality_context.csv'))

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
combined_pval_df = pd.DataFrame(combined_pval_list)
mean_causality['p_value'] = combined_pval_df['combined_p_value']

# %%
f, axs = plt.subplots(1, 3, figsize=(2.5*3, 2.5), dpi=dpi)
for i, obj in enumerate(objects):
    matrix = mean_causality[mean_causality['object'] == obj].pivot(
        index='region1', columns='region2', values='f_stat'
    )
    sns.heatmap(matrix, annot=False, cmap='viridis', vmin=1, vmax=None, square=True, ax=axs[i])
    axs[i].set(title=f'{obj}', xlabel='Target region', ylabel='Source region')
 
plt.tight_layout()
plt.show()

# %%

for obj in ['object1', 'object2', 'object3']:
    plt.figure(figsize=(3, 3), dpi=300)
    
    # Filter your DataFrame
    df = mean_causality[mean_causality['object'] == obj].copy()
    df = df[df['p_value'] < 0.05]  
    #df = df[df['f_stat'] > 1]  
    
    # Build a directed graph
    G = nx.DiGraph()
    
    # Add weighted edges
    for _, row in df.iterrows():
        G.add_edge(row['region1'], row['region2'], weight=row['f_stat'])
    
    # Add all expected nodes explicitly to ensure isolated ones are included
    node_order = ['VIS', 'AUD', 'TEa', 'PERI 36', 'PERI 35', 'LEC', 'dCA1', 'iCA1']
    G.add_nodes_from(node_order)
    
    # Layout
    pos = nx.circular_layout(dict(zip(node_order, range(len(node_order)))))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=7)
    
    # Draw edges with weights (causality strength)
    edges = G.edges(data=True)
    weights = [d['weight'] for (_, _, d) in edges]
    scaled_weights = [0.75 + 1.5 * ((w - min(weights)) / (max(weights) - min(weights))) for w in weights]
    
    nx.draw_networkx_edges(
        G, pos, edgelist=edges,
        width=scaled_weights,
        min_source_margin=18,
        min_target_margin=18,
        arrows=True,
        arrowsize=12,
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
    plt.savefig(join(path_dict['google_drive_fig_path'], f'granger_causality_{obj}.jpg'), dpi=600)