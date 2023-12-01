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
BIN_SIZE = 0.5  # for the other variables
FREQ = [5, 10]
N_STATES_SELECT = np.arange(2, 11)
N_STATES = 5

# Get paths
path_dict = paths(sync=False)
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
    print(f'Starting state {s} of {N_STATES_SELECT[-1]}')

    # Cross validate
    this_ll = np.empty(K_FOLDS)
    for f, (train_index, test_index) in enumerate(kf.split(behav_signals)):
        
        # Fit HMM on training data
        simple_hmm = ssm.HMM(s, behav_signals.shape[1], observations='gaussian')
        lls = simple_hmm.fit(behav_signals[train_index, :], method='em',
                             transitions='sticky')

        # Get log-likelihood on test data
        this_ll[f] = simple_hmm.log_likelihood(behav_signals[test_index, :])
    log_likelihood[j] = np.mean(this_ll)

# %% Plot figure

colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.75, 1.75), dpi=dpi)
ax1.plot(N_STATES_SELECT, log_likelihood, marker='o')
ax1.set(xlabel='Number of states', ylabel='Log-likelihood', xticks=N_STATES_SELECT,
        yticks=[-3000, 3000])
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(path_dict['fig_path'], 'hmm_behavior_n_states.jpg'), dpi=600)
