# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 09:46:06 2025

By Guido Meijer
"""


import os
from os.path import join, isdir, isfile
import numpy as np
import pandas as pd
from glob import glob
from scipy.ndimage import gaussian_filter1d
from itertools import chain
from logreader import create_bp_structure, compute_onsets, compute_offsets
import matplotlib.pyplot as plt
from msvr_functions import paths, figure_style
colors, dpi = figure_style()


root = r'V:\imaging1\guido\Subjects\478153\20250619'
data_files = glob(join(root, 'raw_behavior_data', '*.b64'))
data_file = data_files[0]   
data = create_bp_structure(data_file)


# %%
# Detect invertions because they happen super close to the environment trigger

jj = 10

last_trial = np.where(np.diff(data['digitalIn'][:, 12]) != 0)[0][-1]  
env_trace = data['digitalIn'][:last_trial, 12].copy()
sound_trace = data['digitalIn'][:last_trial, jj].copy()
fixed_trace = data['digitalIn'][:last_trial, jj].copy()
sound_toggles = np.where(np.diff(fixed_trace) != 0)[0]
env_toggles = np.where(np.diff(env_trace) != 0)[0]
sound_env = np.array([np.min(np.abs(i - env_toggles)) for i in sound_toggles])
inv_inds = sound_toggles[sound_env < 10]

# Loop over all invertions and invert the trace after the invertion back
for inv in inv_inds:
    print(f'Channel {jj} inverted mid-session')
    fixed_trace[inv + 1:] = 1 - fixed_trace[inv + 1:]

# Sometimes a random inversion occurs in a trial
while(np.sum((fixed_trace == 1) & (env_trace == 1))) > 0:
    
    # Detect where it goes wrong (both traces are positive which should not happen)
    # and invert from the last toggle before where it went wrong
    went_wrong = np.where(np.diff((fixed_trace == 1) & (env_trace == 1)) != 0)[0][0]
    sound_toggles = np.where(np.diff(fixed_trace) != 0)[0]
    inv_ind = sound_toggles[np.where(went_wrong - sound_toggles > 0)[0][-1]]
    fixed_trace[inv_ind + 1:] = 1 - fixed_trace[inv_ind + 1:]

# Check whether it worked 
if np.sum(fixed_trace[env_trace == 1]) / np.sum(env_trace == 1) > 0.1:
    print('Inversion patching failed')
    
# %% Plot

for inv in inv_inds:
    f, ax = plt.subplots(1, 1, figsize=(7, 3), dpi=dpi)
    ax.plot(env_trace[inv - 50000 : inv + 50000], lw=1)
    ax.plot(sound_trace[inv - 50000 : inv + 50000], lw=1, ls='--')
    ax.plot(fixed_trace[inv - 50000 : inv + 50000], lw=1, ls='dotted')
    ax.scatter(50000, 1.05, marker='*', color='r')
    
f, ax = plt.subplots(1, 1, figsize=(7, 3), dpi=dpi)
ax.plot(env_trace)
ax.plot(fixed_trace, ls='--', color='g')
#ax.plot(data['digitalIn'][:last_trial, 9], ls='--', color='m')