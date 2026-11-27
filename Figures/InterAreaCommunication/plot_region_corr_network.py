# %% -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 2026

Plot the noise correlations of the fixed-axis LDA projections between brain regions (output of
region_corr_lda.py) as a network: the regions are placed on a circle and are connected by lines
whose width and color reflect the correlation.

The correlation per region pair is quantified as the area under the curve (AUC) of the
correlation over time, separately for the pre-entry (-1 to 0 s) and post-entry (0 to 1 s) epoch
and separately per object. The AUC is first computed per session and then averaged over
sessions. The epochs follow the LDA epochs of region_corr_lda.py: the bin at t = 0 belongs to the
pre-entry epoch.

Per region pair, the session AUCs are tested against zero with a one-sample t-test. The p-values
are corrected for multiple comparisons over region pairs with the Benjamini-Hochberg false
discovery rate (FDR), separately per object x epoch combination. With ONLY_SIGNIFICANT = True
only the region pairs with an FDR-corrected p < ALPHA are drawn.

By Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from scipy import stats
from scipy.stats import false_discovery_control
from msvr_functions import paths, figure_style

colors, dpi = figure_style()

# Settings
BIN_SIZE = 0.2  # s, has to match region_corr_lda.py
STEP_SIZE = 0.05  # s, has to match region_corr_lda.py
EPOCHS = {'Pre-entry (-1 to 0 s)': [-0.5, 0],
          'Post-entry (0 to 1 s)': [0.2, 1]}
MIN_COVERAGE = 0.8  # minimum fraction of time bins in an epoch that need a correlation value
MIN_SESSIONS = 3  # minimum number of sessions for a region pair to be plotted
ONLY_SIGNIFICANT = True  # only draw region pairs with a session AUC significantly different from 0
ALPHA = 0.05
NODE_ORDER = ['VIS', 'AUD', 'TEa', 'PERI', 'LEC', 'CA1']  # regions not in this list are appended
CMAP = 'coolwarm'

# Load in data
path_dict = paths()
suffix = f'_bin{int(round(BIN_SIZE * 1000))}ms_step{int(round(STEP_SIZE * 1000))}ms'
corr_df = pd.read_csv(path_dict['google_drive_data_path'] / f'region_corr_lda_fixed_axis{suffix}.csv')
corr_df['object'] = corr_df['object'].astype(str)
trapezoid = getattr(np, 'trapezoid', None) or np.trapz

# Assign every time bin to an epoch: the first epoch is [start, end], later epochs (start, end]
epoch_masks = {}
for i, (epoch_name, (start, end)) in enumerate(EPOCHS.items()):
    in_start = (corr_df['time'] >= start - 1e-9) if i == 0 else (corr_df['time'] > start + 1e-9)
    epoch_masks[epoch_name] = in_start & (corr_df['time'] <= end + 1e-9)
n_bins_epoch = {epoch_name: np.unique(corr_df.loc[mask, 'time']).shape[0]
                for epoch_name, mask in epoch_masks.items()}

# Area under the curve per session, region pair, object and epoch
auc_results = []
for epoch_name, mask in epoch_masks.items():
    for (subject, date, region1, region2, obj), this_df in corr_df[mask].groupby(
            ['subject', 'date', 'region1', 'region2', 'object']):
        this_df = this_df.dropna(subset=['r']).sort_values('time')
        if this_df.shape[0] < MIN_COVERAGE * n_bins_epoch[epoch_name]:
            continue
        auc_results.append({
            'subject': subject, 'date': date, 'region1': region1, 'region2': region2,
            'object': obj, 'epoch': epoch_name,
            'auc': trapezoid(this_df['r'].values, this_df['time'].values)})
auc_df = pd.DataFrame(auc_results)

# Average over sessions and test against zero
summary_results = []
for (region1, region2, obj, epoch_name), this_df in auc_df.groupby(
        ['region1', 'region2', 'object', 'epoch']):
    if this_df.shape[0] < MIN_SESSIONS:
        continue
    summary_results.append({
        'region1': region1, 'region2': region2, 'object': obj, 'epoch': epoch_name,
        'auc': this_df['auc'].mean(), 'n_sessions': this_df.shape[0],
        'p_value': stats.ttest_1samp(this_df['auc'], 0).pvalue})
summary_df = pd.DataFrame(summary_results)

# Benjamini-Hochberg FDR correction over region pairs, per object x epoch combination
summary_df['p_fdr'] = np.nan
for _, this_df in summary_df.groupby(['object', 'epoch']):
    valid = this_df['p_value'].notna()
    summary_df.loc[this_df.index[valid], 'p_fdr'] = false_discovery_control(
        this_df.loc[valid, 'p_value'], method='bh')

# Node order on the circle
all_regions = np.unique(np.concatenate([summary_df['region1'], summary_df['region2']]))
node_order = ([r for r in NODE_ORDER if r in all_regions]
              + [r for r in all_regions if r not in NODE_ORDER])
pos = nx.circular_layout(dict(zip(node_order, range(len(node_order)))))

# Symmetric color scale and edge width scaling shared across all panels
max_auc = summary_df['auc'].abs().max()
norm = mpl.colors.Normalize(vmin=-max_auc, vmax=max_auc)
cmap = plt.get_cmap(CMAP)

# %% Plot network per epoch, one figure per object (same color scale for all figures)

for obj in sorted(summary_df['object'].unique()):
    f, axs = plt.subplots(1, len(EPOCHS), figsize=(4.8, 2.5), dpi=dpi,
                          squeeze=False)
    axs = axs[0]
    for j, epoch_name in enumerate(EPOCHS.keys()):
        ax = axs[j]
        df = summary_df[(summary_df['object'] == obj) & (summary_df['epoch'] == epoch_name)]
        if ONLY_SIGNIFICANT:
            df = df[df['p_fdr'] < ALPHA]

        # Build an undirected graph (correlations are symmetric)
        G = nx.Graph()
        G.add_nodes_from(node_order)
        for row in df.itertuples():
            G.add_edge(row.region1, row.region2, weight=row.auc)

        # Draw nodes and labels
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color='lightgrey')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

        # Draw edges: width scales with |AUC|, color with the sign and size of the AUC
        edges = list(G.edges(data=True))
        if len(edges) > 0:
            weights = np.array([d['weight'] for (_, _, d) in edges])
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(u, v) for (u, v, _) in edges],
                width=0.5 + 3 * np.abs(weights) / max_auc,
                edge_color=[cmap(norm(w)) for w in weights],
                node_size=500)

        x_values, y_values = zip(*pos.values())
        ax.set(xlim=(min(x_values) - 0.25, max(x_values) + 0.25),
               ylim=(min(y_values) - 0.25, max(y_values) + 0.25))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(epoch_name, fontsize=7)

    # Colorbar
    f.subplots_adjust(right=0.85)
    cax = f.add_axes([0.88, 0.25, 0.01, 0.5])
    cbar = f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label('Correlation AUC (r·s)', fontsize=7, rotation=270, labelpad=10)
    cbar.set_ticks([-0.1, 0, 0.1])
    cbar.set_ticklabels([-0.1, 0, 0.1])
    cbar.ax.tick_params(labelsize=7)

    plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / f'region_corr_lda_network_object{obj}.pdf')
    plt.show()
