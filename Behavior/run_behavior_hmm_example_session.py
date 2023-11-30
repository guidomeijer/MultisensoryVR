# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:16:35 2023 by Guido Meijer
"""

import ssm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from scipy.signal import spectrogram
from sklearn.model_selection import KFold
import pandas as pd
from msvr_functions import paths, figure_style, bandpass_filter, bin_signal

# Example session
SUBJECT = '452506'
SESSION = '20231124'
LICK_DUR = 0.1
K_FOLDS = 10
WIN_SIZE = 1  # s for breathing power
BIN_SIZE = 0.25 # for the other variables
FREQ = [5, 10]
N_STATES_SELECT = np.arange(2, 11)
N_STATES = 3

# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Load in data
trials_df = pd.read_csv(join(data_path, 'Subjects', SUBJECT, SESSION, 'trials.csv'))
pupil_df = pd.read_csv(join(data_path, 'Subjects', SUBJECT, SESSION, 'pupil.csv'))
camera_times = np.load(join(data_path, 'Subjects', SUBJECT, SESSION, 'camera.times.npy'))
lick_times = np.load(join(data_path, 'Subjects', SUBJECT, SESSION, 'lick.times.npy'))
cont_times = np.load(join(data_path, 'Subjects', SUBJECT, SESSION, 'continuous.times.npy'))
wheel_speed = np.load(join(data_path, 'Subjects', SUBJECT, SESSION, 'continuous.wheelSpeed.npy'))
breathing = np.load(join(data_path, 'Subjects', SUBJECT, SESSION, 'continuous.breathing.npy'))

# Match timestamps and video frames for pupil
pupil_size = pupil_df['width_smooth'].values[:camera_times.shape[0]]

# Use the first and last camera timestamp as the start and end of the other signals
breathing = breathing[(cont_times >= camera_times[0]) & (cont_times <= camera_times[-1])]
wheel_speed = wheel_speed[(cont_times >= camera_times[0]) & (cont_times <= camera_times[-1])]
cont_times = cont_times[(cont_times >= camera_times[0]) & (cont_times <= camera_times[-1])]

# Get sampling rate
cont_sr = int(np.round(1/np.mean(np.diff(cont_times))))

# Create continuous variable from lick times
lick_n_samp = int(np.round(LICK_DUR / (1/cont_sr)))
lick_cont = np.zeros(cont_times.shape[0]).astype(int)
for i, this_lick in enumerate(lick_times):
    this_ind = np.where(cont_times == this_lick)[0][0]
    lick_cont[this_ind:this_ind+lick_n_samp] = 1
    
# Filter breathing trace
breathing = breathing - np.mean(breathing)
breathing_filt = bandpass_filter(breathing, 1, 12, cont_sr, order=1)

# Get breathing power
freq, spec_time, spec = spectrogram(breathing_filt, fs=cont_sr,
                                    nperseg=WIN_SIZE*cont_sr,
                                    noverlap=(WIN_SIZE*cont_sr)-(BIN_SIZE*cont_sr))
spec_time = spec_time + camera_times[0]
breathing_power = np.mean(spec[(freq >= FREQ[0]) & (freq <= FREQ[1]), :], axis=0)
breathing_power = breathing_power[spec_time <= cont_times[-1]]
spec_time = spec_time[spec_time <= cont_times[-1]]

# Use time binning of breathing spectogram to bin the other variables accordingly
bin_edges = np.append(spec_time - (BIN_SIZE / 2), spec_time[-1] + (BIN_SIZE / 2))
running_binned = bin_signal(cont_times, wheel_speed, bin_edges)
lick_binned = bin_signal(cont_times, lick_cont, bin_edges)
pupil_binned = bin_signal(camera_times, pupil_size, bin_edges)

# Construct 2D array 
behav_signals = np.vstack((breathing_power, running_binned, lick_binned, pupil_binned)).T

# Fit HMM
# Loop over different number of states
kf = KFold(n_splits=K_FOLDS, shuffle=False)
log_likelihood = np.empty(N_STATES_SELECT.shape[0])
for j, s in enumerate(N_STATES_SELECT):
    print(f'Starting state {s} of {N_STATES[-1]}')

    # Cross validate
    train_index, test_index = next(kf.split(behav_signals))

    # Fit HMM on training data
    simple_hmm = ssm.HMM(s, behav_signals.shape[1], observations='gaussian')
    lls = simple_hmm.fit(behav_signals[train_index, :], method='em',
                         transitions='sticky')

    # Get log-likelihood on test data
    log_likelihood[j] = simple_hmm.log_likelihood(behav_signals[test_index, :])

# Run final HMM
simple_hmm = ssm.HMM(N_STATES, behav_signals.shape[1], observations='gaussian')
lls = simple_hmm.fit(behav_signals, method='em', transitions='sticky')
zhat = simple_hmm.most_likely_states(behav_signals)

# Normalize all traces between 0 and 1 for plotting
pupil_norm = (pupil_size - np.min(pupil_size)) / np.ptp(pupil_size)
wheel_speed_norm = (wheel_speed - np.min(wheel_speed)) / np.ptp(wheel_speed)
breathing_norm = (breathing_filt - np.min(breathing_filt)) / np.ptp(breathing_filt)
pupil_binned_norm = (pupil_binned - np.min(pupil_binned)) / np.ptp(pupil_binned)
running_binned_norm = (running_binned - np.min(running_binned)) / np.ptp(running_binned)
breathing_power_norm = (breathing_power - np.min(breathing_power)) / np.ptp(breathing_power)

# %% Plot all traces
pupil_norm = pupil_norm[:camera_times.shape[0]]
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(4, 4), dpi=dpi)
ax1.plot(cont_times, lick_cont, color='seagreen')
ax1.plot(camera_times, ((pupil_norm - 0.5)*3) + 2, color='darkmagenta')
ax1.plot(cont_times, ((breathing_norm - 0.5)*10) - 3, color='darkorange')
ax1.plot(cont_times, ((wheel_speed_norm - 0.5)*3) + 5, color='royalblue')
ax1.set(xlim=[440, 455], ylim=[-3.5, 6])
ax1.axis('off')
plt.gcf().text(0.85, 0.325, 'Breathing', fontsize=7, color='darkorange')
plt.gcf().text(0.85, 0.42, 'Licking', fontsize=7, color='seagreen')
plt.gcf().text(0.85, 0.62, 'Pupil', fontsize=7, color='darkmagenta')
plt.gcf().text(0.85, 0.8, 'Running', fontsize=7, color='royalblue')

f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.plot(N_STATES_SELECT, log_likelihood, marker='o')
ax1.set(xlabel='Number of states', ylabel='Log-likelihood', xticks=N_STATES_SELECT)

# %%
f, ax1 = plt.subplots(figsize=(4, 4), dpi=dpi)
ax1.imshow(zhat[None,:], aspect='auto', cmap='Set2', 
          extent=(0, spec_time[-1], -3, 7))
ax1.plot(spec_time, lick_binned, color='seagreen')
ax1.plot(spec_time, ((pupil_binned_norm - 0.5)*3) + 2, color='darkmagenta')
ax1.plot(spec_time, ((breathing_power_norm - 0.5)*5), color='darkorange')
ax1.plot(spec_time, ((running_binned_norm - 0.5)*3) + 5, color='royalblue')


plt.tight_layout()

