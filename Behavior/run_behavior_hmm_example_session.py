# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:16:35 2023 by Guido Meijer
"""

import ssm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from os.path import join
from sklearn.model_selection import KFold
import pandas as pd
from msvr_functions import load_subjects, paths, figure_style, bandpass_filter

# Example session
SUBJECT = '452506'
SESSION = '20231124'
LICK_DUR = 0.1
K_FOLDS = 10
N_STATES = np.arange(2, 21)

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
    
# Fit HMM
kf = KFold(n_splits=K_FOLDS, shuffle=False)

# Loop over different number of states
log_likelihood = np.empty(N_STATES.shape[0])
for j, s in enumerate(N_STATES):
    print(f'Starting state {s} of {N_STATES[-1]}')

    # Cross validate
    train_index, test_index = next(kf.split(binned_spikes))

    # Fit HMM on training data
    simple_hmm = ssm.HMM(s, binned_spikes.shape[1], observations='poisson')
    lls = simple_hmm.fit(binned_spikes[train_index, :], method='em',
                         transitions='sticky')

    # Get log-likelihood on test data
    log_likelihood[j] = simple_hmm.log_likelihood(binned_spikes[test_index, :])



# Normalize all traces between 0 and 1 for plotting
pupil_norm = (pupil_df['width_smooth'] - np.min(pupil_df['width_smooth'])) / np.ptp(pupil_df['width_smooth'])
wheel_speed_norm = (wheel_speed - np.min(wheel_speed)) / np.ptp(wheel_speed)
breathing_norm = (breathing_filt - np.min(breathing_filt)) / np.ptp(breathing_filt)
wheel_speed_norm = (wheel_speed - np.min(wheel_speed)) / np.ptp(wheel_speed)

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

