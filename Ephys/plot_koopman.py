# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:28:53 2025

By Guido Meijer
"""

import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from msvr_functions import paths

# Load in results
path_dict = paths()
causality_df = pd.read_csv(path_dict['save_path'] / 'causality_koopman.csv')

# Get mean causality
mean_causality = causality_df[['region_1', 'region_2', 'causality_score', 'object']].groupby(
    ['region_1', 'region_2', 'object']).mean().reset_index()


for obj in [1, 2]:
    plt.figure(figsize=(3, 3), dpi=300)

    # Filter your DataFrame
    df = mean_causality[mean_causality['object'] == obj].copy()
    df = df[df['causality_score'] > 0.2]

    # Build a directed graph
    G = nx.DiGraph()

    # Add weighted edges
    for _, row in df.iterrows():
        G.add_edge(row['region_1'], row['region_2'], weight=row['causality_score'])

    # Add all expected nodes explicitly to ensure isolated ones are included
    node_order = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']
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