# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 10:45:57 2025

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, isdir, isfile
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.ticker import FormatStrFormatter
from msvr_functions import (paths, peri_multiple_events_time_histogram, load_objects, load_trials,
                            load_neural_data, figure_style)
path_dict = paths()
colors, dpi = figure_style()

# Settings
"""
SUBJECT = '459601'
DATE = '20240409'
PROBE = 'probe00'
NEURON_IDS = [302, 281]
"""

SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
NEURON_IDS = [481, 505]

# Plot settings
MIN_SPEED = 20  # mm/s
T_BEFORE = 2  # s
T_AFTER = 2
T_BIN_SIZE = 0.025
T_SMOOTHING = 0.1

# LDA settings (match region_corr_lda.py)
LDA_OBJECT = 2  # object to use for the LDA example
AXIS_WINDOW = [-2, 0]  # s around object entry, firing rate in this window is used
SHRINKAGE = 'auto'
N_SHOW_SCORES = 8  # number of LDA scores to list per context

# %% Plot object neuron in visual cortex
session_path = path_dict['local_data_path'] / 'Subjects' / SUBJECT / DATE
spikes, clusters, channels = load_neural_data(session_path, PROBE, histology=True, only_good=True)
trials = load_trials(SUBJECT, DATE)
all_obj_df = load_objects(SUBJECT, DATE)
wheel_speed = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.wheelSpeed.npy'))
wheel_dist = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.wheelDistance.npy'))
wheel_times = np.load(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'continuous.times.npy'))
    
# Set a speed threshold, find for each spike its corresponding speed
indices = np.searchsorted(wheel_times, spikes['times'], side='right') - 1
indices = np.clip(indices, 0, wheel_dist.shape[0] - 1)
spike_speed = wheel_speed[indices]
spikes_times = spikes['times'][spike_speed >= MIN_SPEED]
these_clusters = spikes['clusters'][spike_speed >= MIN_SPEED]

# %% Plot centered on object
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 1.75), dpi=dpi, sharey=True)

peri_multiple_events_time_histogram(
    spikes_times, these_clusters,
    all_obj_df.loc[all_obj_df['object'] == 2, 'times'],
    all_obj_df.loc[all_obj_df['object'] == 2, 'goal'],
    [NEURON_IDS[0]], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax1,
    smoothing=T_SMOOTHING, ylim=5,
    pethline_kwargs=[{'color': colors['context1'], 'lw': 1}, {'color': colors['context2'], 'lw': 1}],
    errbar_kwargs=[{'color': colors['context1'], 'alpha': 0.3, 'lw': 0},
                {'color': colors['context2'], 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': colors['context1'], 'lw': 0.5}, {'color': colors['context2'], 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=True)
ax1.set(ylabel='Firing rate (spks/s)', yticks=[0, 5], xlabel='', xticks=[-T_BEFORE, 0, T_AFTER],
        title='TEa neuron 1')
ax1.yaxis.set_label_coords(-0.25, 0.75)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

peri_multiple_events_time_histogram(
    spikes_times, these_clusters,
    all_obj_df.loc[all_obj_df['object'] == 2, 'times'],
    all_obj_df.loc[all_obj_df['object'] == 2, 'goal'],
    [NEURON_IDS[1]], t_before=T_BEFORE, t_after=T_AFTER, bin_size=T_BIN_SIZE, ax=ax2,
    smoothing=T_SMOOTHING, ylim=5,
    pethline_kwargs=[{'color': colors['context1'], 'lw': 1}, {'color': colors['context2'], 'lw': 1}],
    errbar_kwargs=[{'color': colors['context1'], 'alpha': 0.3, 'lw': 0},
                {'color': colors['context2'], 'alpha': 0.3, 'lw': 0}],
    raster_kwargs=[{'color': colors['context1'], 'lw': 0.5}, {'color': colors['context2'], 'lw': 0.5}],
    eventline_kwargs={'lw': 0}, include_raster=True)
ax2.set(ylabel='', yticks=[0, 5], xlabel='', xticks=[-T_BEFORE, 0, T_AFTER], title='TEa neuron 2')
ax2.yaxis.set_label_coords(-0.25, 0.75)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

f.supxlabel('Time from second rewarded object (s)', x=0.55, y=0.02)

plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15, right=0.97)
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'example_neurons.pdf')
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'example_neurons.jpg', dpi=600)
plt.show()

# %% LDA projection of the two example neurons

def lda_projection(model, X):
    """
    Project X onto the unit-length LDA axis, centered on the midpoint between the class means.
    With equal priors the decision function is w.x - w.(mu_A + mu_B) / 2, so dividing it by |w|
    gives the distance along the discriminant axis (positive = second class).
    """
    return model.decision_function(X) / np.linalg.norm(model.coef_[0])


# Firing rate of both neurons in the axis window for every entry of the object
obj_df = all_obj_df[(all_obj_df['object'] == LDA_OBJECT) & ~np.isnan(all_obj_df['times'])]
obj_df = obj_df.reset_index(drop=True)
X = np.empty((obj_df.shape[0], len(NEURON_IDS)))
for n, neuron_id in enumerate(NEURON_IDS):
    neuron_spikes = spikes_times[these_clusters == neuron_id]
    for e, entry_time in enumerate(obj_df['times']):
        X[e, n] = np.sum((neuron_spikes > entry_time + AXIS_WINDOW[0])
                         & (neuron_spikes <= entry_time + AXIS_WINDOW[1]))
X = X / (AXIS_WINDOW[1] - AXIS_WINDOW[0])

# z-score per neuron (as in region_corr_lda.py)
#X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = obj_df['sound'].values.astype(int)

# Which context is the goal context for this object (for the colors)
goal_context = obj_df.loc[obj_df['goal'] == 1, 'sound'].values[0]
context_colors = {c: colors['context1'] if c == goal_context else colors['context2'] for c in [1, 2]}
context_labels = {c: 'Context 1' if c == goal_context else 'Context 2' for c in [1, 2]}

# Fit LDA with the same settings as in the analysis (equal priors)
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=SHRINKAGE, priors=[0.5, 0.5])
lda.fit(X, y)
scores = lda_projection(lda, X)
w_unit = lda.coef_[0] / np.linalg.norm(lda.coef_[0])
midpoint = lda.means_.mean(axis=0)

# %% Plot LDA projection

f, ax = plt.subplots(1, 1, figsize=(1.8, 1.8), dpi=dpi)

# Single entries
for c in [1, 2]:
    ax.scatter(*X[y == c].T, s=8, color=context_colors[c], alpha=0.5, lw=0,
               label=context_labels[c])

# LDA axis through the midpoint
lim = np.ceil(np.max(np.abs(X)))
ax.plot(*(midpoint[:, None] + np.array([-2 * lim, 2 * lim]) * w_unit[:, None]), color='k',
        lw=1, zorder=1)
ax.plot(*midpoint, marker='+', color='k', ms=5, zorder=3)  # score = 0

# Decision boundary (score = 0), perpendicular to the axis
w_perp = np.array([-w_unit[1], w_unit[0]])
ax.plot(*(midpoint[:, None] + np.array([-2 * lim, 2 * lim]) * w_perp[:, None]), color='grey',
        lw=0.5, ls='--', zorder=0)

ax.set(xlabel=f'Neuron 2 firing rate (spks/s)',
       ylabel=f'Neuron 1 firing rate (spks/s)',
       xlim=[-0.5, 7.1], ylim=[-0.5, 7.1], aspect='equal',
       xticks=[0, 1, 2, 3, 4, 5, 6, 7], yticks=[0, 1, 2, 3, 4, 5, 6, 7])
#ax.legend(frameon=False, loc='upper left', fontsize=6, handletextpad=0.2)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'example_neurons_lda.pdf')
plt.show()

# %% Plot LDA scores of context A and B as two vertical lists of numbers

# Context A = the context labeled 'Context 1', context B = 'Context 2'
context_order = sorted([1, 2], key=lambda c: context_labels[c])
f, ax = plt.subplots(1, 1, figsize=(1.2, 1.8), dpi=dpi)
row_height = 1 / (N_SHOW_SCORES + 2)
for col, (c, name) in enumerate(zip(context_order, ['Context A', 'Context B'])):
    x = 0.25 + 0.5 * col
    ax.text(x, 1, name, ha='center', va='top', fontsize=6, color=context_colors[c],
            fontweight='bold')
    # Only show correctly classified entries: positive score = lda.classes_[1]
    correct_sign = 1 if c == lda.classes_[1] else -1
    c_scores = scores[(y == c) & (np.sign(scores) == correct_sign)]
    for i, score in enumerate(c_scores[:N_SHOW_SCORES]):
        ax.text(x, 1 - (i + 1.3) * row_height, f'{score:.2f}', ha='center', va='top',
                fontsize=6, color=context_colors[c])
    if c_scores.shape[0] > N_SHOW_SCORES:
        # Vertical ellipsis as three dots (the unicode character is not in the font)
        y_dots = 1 - (N_SHOW_SCORES + 1.5) * row_height - np.arange(3) * row_height * 0.25
        ax.plot(np.full(3, x), y_dots, ls='', marker='o', ms=0.5, color=context_colors[c])
ax.set(xlim=[0, 1], ylim=[-0.1, 1])
ax.axis('off')
plt.tight_layout()
# plt.savefig(path_dict['paper_fig_path'] / 'InterAreaCommunication' / 'example_neurons_lda_scores.pdf')
plt.show()
