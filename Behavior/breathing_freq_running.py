# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:00:55 2023 by Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from scipy.signal import spectrogram, butter, filtfilt
from msvr_functions import paths, figure_style

# Settings
SUBJECT = '450408'
DATE = '20230929'
WIN_SIZE = 2  # s
WIN_SHIFT = 4  # s
FS = 1000  # sampling rate
FREQ = [3, 10]
RUNNING_BINS = np.arange(0, 101, 10)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Load in data
breathing = np.load(
    join(data_path, 'Subjects', f'{SUBJECT}', f'{DATE}', 'continuous.breathing.npy'))
running = np.load(
    join(data_path, 'Subjects', f'{SUBJECT}', f'{DATE}', 'continuous.wheelSpeed.npy'))
timestamps = np.load(join(data_path, 'Subjects', f'{SUBJECT}', f'{DATE}', 'continuous.times.npy'))
acceleration = np.diff(running)

# Get breathing power
# breathing = breathing - np.mean(breathing)
# breathing_filt = butter_bandpass_filter(breathing, 2, 50, int(1/np.diff(timestamps)[0]), order=1)
freq, time_bins, spec = spectrogram(breathing, fs=FS,
                                    nperseg=WIN_SIZE*FS,
                                    noverlap=(WIN_SIZE*FS)-(WIN_SHIFT*FS))

# Get running speed and frequency per time bin
mean_running = np.empty(time_bins.shape[0])
mean_acc = np.empty(time_bins.shape[0])
breathing_freq = np.empty(time_bins.shape[0])
breathing_amp = np.empty(time_bins.shape[0])
for i, bin_center in enumerate(time_bins):
    if np.mod(i, 1000) == 0:
        print(f'Timebin {i} of {len(time_bins)}')
    mean_running[i] = np.median(running[(timestamps >= bin_center - WIN_SIZE/2)
                                        & (timestamps <= bin_center + WIN_SIZE/2)])
    mean_acc[i] = np.median(acceleration[(timestamps[:-1] >= bin_center - WIN_SIZE/2)
                                         & (timestamps[:-1] <= bin_center + WIN_SIZE/2)])
    breathing_freq[i] = freq[np.argmax(spec[:, i])]
    breathing_amp[i] = np.mean(spec[(freq >= FREQ[0]) & (freq <= FREQ[1]), i])

# Put in dataframe for plotting
running_breathing_df = pd.DataFrame(data={'running': mean_running, 'breathing_freq': breathing_freq,
                                          'acceleration': mean_acc, 'breathing_amp': breathing_amp,
                                          'time': bin_center})

# Exclude very low breathing frequencies
running_breathing_df = running_breathing_df[running_breathing_df['breathing_freq'] > 1]

# Bin running speed
running_breathing_df['running_bins'] = pd.cut(running_breathing_df['running'], RUNNING_BINS,
                                              include_lowest=True,
                                              labels=RUNNING_BINS[:-1]+np.diff(RUNNING_BINS)[0]/2).astype(int)

# %% Plot
colors, dpi = figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)

sns.pointplot(data=running_breathing_df, x='running_bins', y='breathing_freq', ax=ax1,
              errorbar='se', markers='none')
ax1.set(ylim=[5, 8], ylabel='Breathing frequency (Hz)', xlabel='Running speed (cm/s)')

sns.pointplot(data=running_breathing_df, x='running_bins', y='breathing_amp', ax=ax2,
              errorbar='se', markers='none')
ax2.set(ylim=[20, 80], ylabel='Breathing amplitude (PSD)', xlabel='Running speed (cm/s)')

plt.tight_layout()
sns.despine(trim=True)


"""
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)

sns.regplot(data=running_breathing_df, x='running', y='breathing_freq', ax=ax1,
            order=3, ci=None,
            scatter_kws={'lw': 0, 'color': 'grey'},
            line_kws={'color': 'red'})
ax1.set(ylim=[0, 15], ylabel='Breathing frequency (Hz)', xlabel='Running speed (cm/s)')

sns.regplot(data=running_breathing_df, x='running', y='breathing_amp', ax=ax2,
            order=3, ci=None,
            scatter_kws={'lw': 0, 'color': 'grey'},
            line_kws={'color': 'red'})
ax2.set(ylim=[0, 100], ylabel='Breathing amplitude (PSD)', xlabel='Running speed (cm/s)')

plt.tight_layout()
sns.despine(trim=True)
"""
