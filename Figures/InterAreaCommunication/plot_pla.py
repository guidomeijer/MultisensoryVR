# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 30/04/2026
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import networkx as nx
from statsmodels.stats.multitest import multipletests
from msvr_functions import paths, figure_style
path_dict = paths()
colors, dpi = figure_style()

# Settings
REGIONS = ['CA1', 'LEC', 'PERI', 'TEa', 'AUD', 'VIS']

# Load in data
pla_n_df = pd.read_csv(path_dict['google_drive_data_path'] / 'pla_n_components.csv')
pla_decoding_df = pd.read_csv(path_dict['google_drive_data_path'] / 'pla_decoding.csv')
pla_df = pd.read_csv(path_dict['google_drive_data_path'] / 'pla.csv')



# %%

for obj in ['obj1', 'obj2', 'obj3']:

    mean_pla_df = pd.DataFrame()
    this_df = pla_df[pla_df['object'] == obj]
    for r, this_pair in enumerate(np.unique(this_df['region_pair'])):

        pvals = this_df.loc[this_df['region_pair'] == this_pair, 'coupling_p'].values
        #_, combined_p = stats.combine_pvalues(pvals, method='fisher')
        combined_p = stats.binomtest(np.sum(pvals < 0.05), pvals.shape[0], 0.05).pvalue
        mean_pla_df = pd.concat((mean_pla_df, pd.DataFrame(index=[mean_pla_df.shape[0]], data={
            'p_value': combined_p, 'coupling': np.mean(this_df.loc[pla_df['region_pair'] == this_pair, 'coupling_r'].values),
            'region_pair': this_pair, 'region1': this_pair.split('-')[0], 'region2': this_pair.split('-')[1]
        })))

    # Adjust p-values for multiple testing
    _, mean_pla_df['p_adj'], _, _ = multipletests(mean_pla_df['p_value'], method='fdr_bh')

    # Select significant pairs
    mean_pla_df = mean_pla_df[mean_pla_df['p_adj'] > 0.05]

    # Create graph object
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(REGIONS)

    # Add edges with weights from mean_pla_df
    for i, row in mean_pla_df.iterrows():
        G.add_edge(row['region1'], row['region2'], weight=row['coupling'])

    # Define positions in a circle
    pos = nx.circular_layout(G)

    # Plotting
    f, ax = plt.subplots(figsize=(2, 2), dpi=dpi)
    weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]
    edge_widths = [w ** 2 for w in weights]

    nx.draw_networkx_nodes(G, pos, node_color='lightgrey', node_size=300, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_family='Arial', ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='k', alpha=0.7, ax=ax)

    ax.axis('off')
    ax.set(title=obj)
    plt.tight_layout()
    plt.show()


# %%
f, axs = plt.subplots(2, 6, figsize=(7, 3), dpi=dpi, sharey=True, sharex=False)
for r, region in enumerate(REGIONS):
    sns.lineplot(data=pla_decoding_df[(pla_decoding_df['region'] == region) & (pla_decoding_df['object'] == 'obj1')],
                 x='time', y='accuracy', ax=axs[0, r], errorbar='se', err_kws={'lw': 0})
    axs[0, r].set(title=region)

    sns.lineplot(data=pla_decoding_df[(pla_decoding_df['region'] == region) & (pla_decoding_df['object'] == 'obj2')],
                 x='time', y='accuracy', ax=axs[1, r], errorbar='se', err_kws={'lw': 0})

sns.despine(trim=True)
plt.tight_layout()
plt.show()

# %%
g = sns.FacetGrid(pla_n_df, col='region_pair', col_wrap=4, height=3)
g.map_dataframe(sns.lineplot, x='n_components', y='r_norm')
g.set_axis_labels('Number of components', 'Normalized R')
g.set(ylim=(0.6, 1))
g.set_titles(col_template='{col_name}')
plt.tight_layout()
plt.show()

# %%
f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
sns.lineplot(data=pla_n_df, x='n_components', y='r', ax=ax, errorbar='se', err_kws={'lw': 0}, color='k', marker='o',
             markeredgecolor='k')
ax.set(ylabel='Correlation coefficient (r)', xlabel='Number of components', xticks=np.arange(1, 11),
       yticks=[0.14, 0.16, 0.18, 0.2, 0.22])

sns.despine(trim=True)
plt.tight_layout()
plt.show()
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'pla_n_components.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'pla_n_components.jpg', dpi=600)