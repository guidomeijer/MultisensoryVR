# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:05:33 2024 by Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from msvr_functions import paths, load_subjects, figure_style
colors, dpi = figure_style()

# Load in data
path_dict = paths()
mi_df = pd.read_csv(join(path_dict['save_path'], 'region_te_150ms-bins.csv'))

# %%  Plot
region_pairs = np.unique(mi_df['region_pair'])
f, axs = plt.subplots(1, len(region_pairs), figsize=(1.75*len(region_pairs), 1.75), dpi=dpi,
                      sharey=True)

f_sound, axs_sound = plt.subplots(
    1, len(region_pairs), figsize=(1.75*len(region_pairs), 1.75), dpi=dpi,
    sharey=True)
for i, region_pair in enumerate(region_pairs):
    
    """
    long_df = pd.melt(mi_df[mi_df['region_pair'] == region_pair],
                      id_vars='time', value_vars=['te_goal_baseline', 'te_distractor_baseline'])
    sns.lineplot(data=long_df, x='time', y='value', hue='variable', ax=axs[i], errorbar='se',
                 legend=None, hue_order=['te_goal_baseline', 'te_distractor_baseline'],
                 palette=[colors['goal'], colors['no-goal']])
    """
    this_df = mi_df[mi_df['region_pair'] == region_pair]
    axs[i].plot(this_df['time'], this_df['te_goal_baseline'], color=colors['goal'])
    axs[i].fill_between(this_df['time'],
                        this_df['te_goal_baseline'] - this_df['te_sem_goal'],
                        this_df['te_goal_baseline'] + this_df['te_sem_goal'],
                        color=colors['goal'], alpha=0.25, lw=0)
    axs[i].plot(this_df['time'], this_df['te_distractor_baseline'], color=colors['no-goal'])
    axs[i].fill_between(this_df['time'],
                        this_df['te_distractor_baseline'] - this_df['te_sem_distractor'],
                        this_df['te_distractor_baseline'] + this_df['te_sem_distractor'],
                        color=colors['no-goal'], alpha=0.25, lw=0)
    
    axs[i].set(title=f'{region_pair}')
    
    axs_sound[i].plot(this_df['time'], this_df['te_sound_baseline'], color=colors['goal'])
    axs_sound[i].fill_between(this_df['time'],
                        this_df['te_sound_baseline'] - this_df['te_sem_sound'],
                        this_df['te_sound_baseline'] + this_df['te_sem_sound'],
                        color=colors['goal'], alpha=0.25, lw=0)
    axs_sound[i].plot(axs_sound[i].get_xlim(), [0, 0], color='grey', ls='--')
    
    axs_sound[i].set(title=f'{region_pair}')
    
#axs[0].set(ylabel='Correlation (r)', yticks=[-0.04, 0, 0.04])
sns.despine(trim=True)
plt.tight_layout()

# %% Plot individual examples

f, axs = plt.subplots(1, 3, figsize=(5.25, 2), dpi=dpi, sharey=True)

this_df = mi_df[mi_df['region_pair'] == 'PERI 36-HPC']
axs[0].plot(this_df['time'], this_df['te_goal_baseline'], color=colors['goal'], label='Goal')
axs[0].fill_between(this_df['time'],
                    this_df['te_goal_baseline'] - this_df['te_sem_goal'],
                    this_df['te_goal_baseline'] + this_df['te_sem_goal'],
                    color=colors['goal'], alpha=0.25, lw=0)
axs[0].plot(this_df['time'], this_df['te_distractor_baseline'], color=colors['no-goal'], label='No goal')
axs[0].fill_between(this_df['time'],
                    this_df['te_distractor_baseline'] - this_df['te_sem_distractor'],
                    this_df['te_distractor_baseline'] + this_df['te_sem_distractor'],
                    color=colors['no-goal'], alpha=0.25, lw=0)
#axs[0].set(xlabel='Time from object entry (s)', ylabel='Pairwise correlation (r)',
#           title='PERI 36 - HPC', ylim=[-0.02, 0.03], yticks=[-0.02, 0, 0.03])
axs[0].legend(prop={'size': 5})

this_df = mi_df[mi_df['region_pair'] == 'PERI 35-HPC']
axs[1].plot(this_df['time'], this_df['te_goal_baseline'], color=colors['goal'])
axs[1].fill_between(this_df['time'],
                    this_df['te_goal_baseline'] - this_df['te_sem_goal'],
                    this_df['te_goal_baseline'] + this_df['te_sem_goal'],
                    color=colors['goal'], alpha=0.25, lw=0)
axs[1].plot(this_df['time'], this_df['te_distractor_baseline'], color=colors['no-goal'])
axs[1].fill_between(this_df['time'],
                    this_df['te_distractor_baseline'] - this_df['te_sem_distractor'],
                    this_df['te_distractor_baseline'] + this_df['te_sem_distractor'],
                    color=colors['no-goal'], alpha=0.25, lw=0)
axs[1].set(xlabel='Time from object entry (s)', title='PERI 35 - HPC')

this_df = mi_df[mi_df['region_pair'] == 'TEa-HPC']
axs[2].plot(this_df['time'], this_df['te_goal_baseline'], color=colors['goal'])
axs[2].fill_between(this_df['time'],
                    this_df['te_goal_baseline'] - this_df['te_sem_goal'],
                    this_df['te_goal_baseline'] + this_df['te_sem_goal'],
                    color=colors['goal'], alpha=0.25, lw=0)
axs[2].plot(this_df['time'], this_df['te_distractor_baseline'], color=colors['no-goal'])
axs[2].fill_between(this_df['time'],
                    this_df['te_distractor_baseline'] - this_df['te_sem_distractor'],
                    this_df['te_distractor_baseline'] + this_df['te_sem_distractor'],
                    color=colors['no-goal'], alpha=0.25, lw=0)
axs[2].set(xlabel='Time from object entry (s)', title='TEa - HPC')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(path_dict['google_drive_fig_path'], 'interarea_te_goal_obj_entry.jpg'), dpi=600)

# %% Plot individual examples
plot_pairs = ['PERI 36-TEa', 'PERI 35-HPC', 'TEa-HPC', 'PERI 36-HPC', 'TEa-VIS']
these_colors = sns.color_palette('Set2')[:5]

f, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=dpi, sharey=True)

for i, region_pair in enumerate(plot_pairs):
    this_df = mi_df[mi_df['region_pair'] == region_pair]
    ax.plot(this_df['time'], this_df['te_sound_baseline'], color=these_colors[i], label=region_pair)
    ax.fill_between(this_df['time'],
                    this_df['te_sound_baseline'] - this_df['te_sem_sound'],
                    this_df['te_sound_baseline'] + this_df['te_sem_sound'],
                    color=these_colors[i], alpha=0.25, lw=0)
   
#ax.set(xlabel='Time from sound onset (s)', ylabel='Pairwise correlation (r)',
#       ylim=[-0.04, 0.04], yticks=[-0.04, 0, 0.04])
ax.legend(prop={'size': 5})



sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(path_dict['google_drive_fig_path'], 'interarea_te_sound_onset.jpg'), dpi=600)

# %%

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = np.unique(mi_df['region_1'])
G.add_nodes_from(nodes)

# Add bidirectional edges with weights
edges = [("A", "B", 0.5), ("B", "A", 0.3), ("A", "C", 1.0), ("C", "A", 0.8), ("B", "D", 0.75), ("D", "B", 0.6)]
for u, v, weight in edges:
    G.add_edge(u, v, weight=weight)

# Draw the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 6))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=700)
nx.draw_networkx_labels(G, pos, font_size=12, font_color="white")

# Draw edges with an offset for bidirectional arrows
for (u, v, d) in G.edges(data=True):
    # Separate the two arrows slightly by adjusting the position
    if G.has_edge(v, u):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrowstyle='->', arrowsize=20, 
                               width=d['weight'] * 2, connectionstyle="arc3,rad=0.2")
    else:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrowstyle='->', arrowsize=20, 
                               width=d['weight'] * 2)

# Display edge weights for bidirectional edges
edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Bidirectional Arrows Between Nodes")
plt.show()