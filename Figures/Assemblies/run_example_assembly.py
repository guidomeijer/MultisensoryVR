# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 24/02/2026
"""
# %%
import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import FastICA
from msvr_functions import (paths, load_neural_data, load_objects,
                            calculate_peths, figure_style, peri_event_trace)
colors, dpi = figure_style()

# Settings
SUBJECT = '462910'
DATE = '20240813'
PROBE = 'probe01'
REGION = 'CA1'
BIN_SIZE = 0.05
MP_THRESHOLD_SCALE = 1.2  # Scale factor for Marchenko-Pastur threshold (higher -> fewer assemblies)
MEMBER_THRESHOLD_Z = 1  # z-score for assembly membership
SMOOTHING = 1

# Initialize
path_dict = paths(sync=False)
rec = pd.read_csv(path_dict['repo_path'] / 'recordings.csv').astype(str)
ripples = pd.read_csv(path_dict['save_path'] / 'ripples.csv')
ripples['subject'] = ripples['subject'].astype(str)
ripples['date'] = ripples['date'].astype(str)
these_ripples = ripples[(ripples['subject'] == SUBJECT) & (ripples['date'] == DATE)]

# %% MAIN

# Load in data
session_path = path_dict['local_data_path'] / 'Subjects' / f'{SUBJECT}' / f'{DATE}'
spikes, clusters, channels = load_neural_data(session_path, PROBE)
trials = pd.read_csv(path_dict['local_data_path'] / 'Subjects' / SUBJECT / DATE / 'trials.csv')
all_obj_df = load_objects(SUBJECT, DATE)

# Get region neurons
region_neurons = clusters['cluster_id'][clusters['region'] == REGION]
region_spikes = spikes['times'][np.isin(spikes['clusters'], region_neurons)]
region_clusters = spikes['clusters'][np.isin(spikes['clusters'], region_neurons)]

# Create binned spike matrix of entire task period
peths_task, binned_spikes = calculate_peths(
    spikes['times'], spikes['clusters'],
    region_neurons, [0],
    pre_time=0, post_time=spikes['times'][-1],
    bin_size=BIN_SIZE, smoothing=0)
binned_spikes = binned_spikes[0]  # (neurons x timebins)
n_timebins = binned_spikes.shape[1]
binned_time = peths_task['tscale']

# Remove silent neurons
active_idx = np.std(binned_spikes, axis=1) > 0
binned_spikes = binned_spikes[active_idx]
n_neurons = binned_spikes.shape[0]

# Z-score spikes
z_spikes = (binned_spikes - np.mean(binned_spikes, axis=1, keepdims=True)) / np.std(binned_spikes, axis=1, keepdims=True)

# Correlation matrix
corr_arr = np.corrcoef(z_spikes)

# Eigen decomposition
evals, evecs = np.linalg.eigh(corr_arr)
idx = evals.argsort()[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

# Marchenko-Pastur threshold
q = n_neurons / n_timebins
lambda_max = ((1 + np.sqrt(q)) ** 2) * MP_THRESHOLD_SCALE

# Number of assemblies
n_assemblies = np.sum(evals > lambda_max)
print(f'Detected {n_assemblies} assemblies')

# PCA projection
pcs = evecs[:, :n_assemblies]
projected_data = pcs.T @ z_spikes

# ICA
ica = FastICA(n_components=n_assemblies, random_state=0)
activations = ica.fit_transform(projected_data.T).T

# Assembly patterns (neurons x assemblies)
assembly_patterns = pcs @ ica.mixing_

# Correct sign (make max absolute weight positive)
for k in range(n_assemblies):
    if np.abs(assembly_patterns[:, k].min()) > np.abs(assembly_patterns[:, k].max()):
        assembly_patterns[:, k] *= -1
        activations[k, :] *= -1


# %% Assign neurons to assemblies
# A neuron is considered a member of an assembly if its weight is MEMBER_THRESHOLD_Z
# standard deviations above the mean of all weights for that assembly.
# We assign each neuron to only one assembly, the one where it has the highest z-scored weight.
neuron_assembly_assignment = -1 * np.ones(n_neurons, dtype=int)  # -1 for no assembly
z_assembly_patterns = np.zeros_like(assembly_patterns)
for k in range(n_assemblies):
    weights = assembly_patterns[:, k]
    z_assembly_patterns[:, k] = (weights - np.mean(weights)) / np.std(weights)

# For each neuron, find the assembly it has the max z-scored weight for
max_z_assembly_idx = np.argmax(z_assembly_patterns, axis=1)

for i in range(n_neurons):
    assembly_idx = max_z_assembly_idx[i]
    if z_assembly_patterns[i, assembly_idx] > MEMBER_THRESHOLD_Z:
        neuron_assembly_assignment[i] = assembly_idx

# Create an ordered list of neurons that are part of an assembly
# Neurons are ordered by assembly, and within each assembly, by weight.
ordered_neurons = np.array([], dtype=int)
assembly_boundaries = [0]
for k in range(n_assemblies):
    # Get members of this assembly
    members = np.where(neuron_assembly_assignment == k)[0]
    if len(members) == 0:
        assembly_boundaries.append(assembly_boundaries[-1])
        continue

    # Sort members by their weight in the assembly pattern
    member_weights = assembly_patterns[members, k]
    sorted_member_indices = np.argsort(member_weights)[::-1]
    ordered_neurons = np.concatenate((ordered_neurons, members[sorted_member_indices]))
    assembly_boundaries.append(len(ordered_neurons))

print(f'Found {len(ordered_neurons)} neurons belonging to {n_assemblies} assemblies.')


# %% Plotting

# --- 1. Eigenvalue spectrum plot ---
plot_n = 20
f, ax = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax.plot(np.arange(1, plot_n+1), evals[:plot_n], marker='o')
ax.plot([0, plot_n], [lambda_max, lambda_max], ls='--', color='red')
ax.text(16, lambda_max + 1, 'MP threshold', color='red', ha='center')
ax.plot([n_assemblies, n_assemblies], [0, lambda_max], ls='--', color='grey')
ax.set(ylabel='Eigenvalue', xlabel='Component',
       xticks=[1, n_assemblies, 10, 20], yticks=[0, np.ceil(evals[0])],
       ylim=[0, np.ceil(evals[0])], xlim=[0.5, plot_n + 0.5])
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / f'n_assemblies_{SUBJECT}_{DATE}_{REGION}.jpg', dpi=600)
plt.show()

# %% --- 2. Example assembly activation plot ---
# Select a time window around an event
t_center = these_ripples['start_times'].values[0]

t_win = [-1, 1]  # seconds around t_center
time_slice = (binned_time >= t_center + t_win[0]) & (binned_time <= t_center + t_win[1])
plot_time = binned_time[time_slice]

# Get the activity of the ordered neurons in this time window
plot_z_spikes = z_spikes[ordered_neurons, :][:, time_slice]

# Get the assembly activations in this time window
plot_activations = activations[:, time_slice]

# Define colors
assembly_colors = sns.color_palette('tab10', n_assemblies)

# Create figure
f, axs = plt.subplots(2, 3, figsize=(3, 2), dpi=dpi,
                      gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [0.03, 1, 0.03], 'wspace': 0.05},
                      sharex='col')
(ax_assembly, ax1, cax), (ax_dummy, ax2, _) = axs
ax_dummy.axis('off')
axs[1, 2].axis('off')

# Plot heatmap of neural activity
im = ax1.imshow(plot_z_spikes, aspect='auto', cmap='coolwarm',
                vmin=-2, vmax=2,  # clip z-scores for better visualization
                extent=[plot_time[0], plot_time[-1], len(ordered_neurons), 0])
ax1.set_ylabel('Neurons', labelpad=15)
ax1.set_yticks([])  # No y-ticks for individual neurons

# Plot assembly identity bar
ax_assembly.set_ylim(len(ordered_neurons), 0)
ax_assembly.set_xlim(0, 1)
ax_assembly.axis('off')
for k in range(n_assemblies):
    rect = plt.Rectangle((0, assembly_boundaries[k]), 1, assembly_boundaries[k+1]-assembly_boundaries[k],
                         color=assembly_colors[k])
    ax_assembly.add_patch(rect)

# Add colorbar for heatmap
cbar = f.colorbar(im, cax=cax, label='Z-scored activity', ticks=[-2, 0, 2])

# Plot assembly activations
for k in range(n_assemblies):
    ax2.plot(plot_time, plot_activations[k, :], lw=1.5, label=f'Assembly {k+1}', color=assembly_colors[k])
ax2.set_xlabel('Time from ripple onset (s)')
ax2.set_ylabel('Assembly\nactivation')
ax2.set(xticks=[t_center-1, t_center-0.5, t_center, t_center+0.5, t_center+1], xticklabels=[-1, -0.5, 0, 0.5, 1])

# Add event marker
ax1.axvline(x=t_center, color='white', linestyle='--')
ax2.axvline(x=t_center, color='grey', linestyle='--')

sns.despine(fig=f)
plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.9)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / f'assembly_activity_{SUBJECT}_{DATE}_{REGION}.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / f'assembly_activity_{SUBJECT}_{DATE}_{REGION}.pdf')
plt.show()

# %%

plot_assembly = 3

f, ax = plt.subplots(figsize=(1.3, 1.8), dpi=dpi)
peri_event_trace(activations[plot_assembly, :], binned_time, these_ripples['start_times'],
                 np.ones(these_ripples.shape[0]), t_before=1, t_after=1, ax=ax,
                 color_palette=[assembly_colors[plot_assembly]])
ax.set(xticks=[-1, 0, 1], xlabel='Time from ripple start (s)', ylabel='Assembly activation', yticks=[0, 1, 2, 3, 4],
       title='Ripples', ylim=[0, 4])

sns.despine(trim=True)
plt.subplots_adjust(left=0.22, right=0.98, top=0.85, bottom=0.21)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / f'assembly_example_ripple_{SUBJECT}_{DATE}_{REGION}.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / f'assembly_example_ripple_{SUBJECT}_{DATE}_{REGION}.pdf')
plt.show()

#%%
# Do some smoothing
activations_smooth = gaussian_filter1d(activations, sigma=SMOOTHING, axis=1)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.2*2, 1.8), dpi=dpi, sharey=True)

peri_event_trace(activations_smooth[plot_assembly, :], binned_time,
                 all_obj_df.loc[all_obj_df['object'] == 1, 'times'],
                 all_obj_df.loc[all_obj_df['object'] == 1, 'goal'].values + 1,
                 t_before=2, t_after=1, ax=ax1,
                 color_palette=[colors['no-goal'], colors['goal']])
ax1.set(xticks=np.arange(-2, 1.5), xlabel='', yticks=[-0.6, -0.1], ylim=[-0.6, -0.1],
        title='Rewarded object 1')
ax1.set_ylabel('Assembly activation', labelpad=-10)

peri_event_trace(activations_smooth[plot_assembly, :], binned_time,
                 all_obj_df.loc[all_obj_df['object'] == 2, 'times'],
                 all_obj_df.loc[all_obj_df['object'] == 2, 'goal'].values + 1,
                 t_before=2, t_after=1, ax=ax2,
                 color_palette=[colors['no-goal'], colors['goal']])
ax2.set(xticks=np.arange(-2, 1.5), xlabel='', title='Rewarded object 2')

f.text(0.5, 0.04, 'Time from object entry (s)', ha='center')
sns.despine(trim=True)
plt.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.21)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / f'assembly_example_hitmis_{SUBJECT}_{DATE}_{REGION}.jpg', dpi=600)
plt.savefig(path_dict['paper_fig_path'] / 'Assemblies' / f'assembly_example_hitmis_{SUBJECT}_{DATE}_{REGION}.pdf')
plt.show()