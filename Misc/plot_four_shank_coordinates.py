# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:31:09 2025

By Guido Meijer
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = Path(r'D:\MultisensoryVR\Subjects\466395\20241115\probe01')

loc_coords = np.load(PATH / 'channels.localCoordinates.npy')
raw_ind = np.load(PATH / 'channels.rawInd.npy')
loc_brain = pd.read_csv(PATH / 'channels.brainLocation.csv')
ap_rms = np.load(PATH / '_iblqc_ephysChannels.apRMS.npy')
ap_rms_labels = np.load(PATH / '_iblqc_ephysChannels.labels.npy')
lf_freqs = np.load(PATH / '_iblqc_ephysSpectralDensityLF.freqs.npy')
lf_power = np.load(PATH / '_iblqc_ephysSpectralDensityLF.power.npy')

ripple_power = np.mean(lf_power[(lf_freqs > 150) & (lf_freqs < 250), :], axis=0)
dg_spike_power = np.mean(lf_power[(lf_freqs > 10) & (lf_freqs < 100), :], axis=0)

# %%

f, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
cax = ax1.scatter(loc_coords[:, 0], loc_coords[:, 1], c=np.arange(loc_coords.shape[0]), cmap='rainbow')
ax1.set(xlabel='lateral um', ylabel='axial um')
plt.colorbar(cax)

f, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
cax = ax1.scatter(loc_brain['x'], loc_brain['z'], c=loc_brain.index, cmap='rainbow')
ax1.set(xlabel='x', ylabel='z')
plt.colorbar(cax)

f, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
cax = ax1.scatter(loc_coords[:, 0], loc_coords[:, 1], c=dg_spike_power[raw_ind], cmap='magma')
ax1.set(xlabel='lateral um', ylabel='axial um')
plt.colorbar(cax)