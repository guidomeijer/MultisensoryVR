# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:00:55 2023 by Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from scipy.signal import butter, filtfilt
from msvr_functions import paths, figure_style


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Load in data
breathing = np.load(join(data_path, 'Subjects', '450409', '20230929', 'continuous.breathing.npy'))
timestamps = np.load(join(data_path, 'Subjects', '450409', '20230929', 'continuous.times.npy'))

# Filter trace
breathing = breathing - np.mean(breathing)
breathing_filt = butter_bandpass_filter(breathing, 1, 12, int(1/np.diff(timestamps)[0]), order=1)

# Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.75, 2), dpi=dpi)
ax1.plot(timestamps[10000:11000], breathing[10000:11000])
# ax1.plot(timestamps[:1000], breathing_filt[:1000])
ax1.set(yticks=[], xlabel='Time (s)')

plt.tight_layout()
sns.despine(trim=True, left=True)
plt.savefig(join(path_dict['fig_path'], 'breathing_example.jpg'), dpi=600)
