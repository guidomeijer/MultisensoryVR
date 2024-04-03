# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:32:45 2023

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from brainbox.plot import peri_event_time_histogram
from msvr_functions import paths, figure_style, load_neural_data
#colors, dpi = figure_style()

# Settings
SUBJECT = '450409'
DATE = '20231012'
PROBE = 'probe00'

# Get paths
path_dict = paths()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters, channels = load_neural_data(session_path, PROBE, only_good=False)


f, ax = plt.subplots()
#a = np.diff(spikes['times'][np.where(np.diff(spikes['samples']) > (0.007 * 30_000))]) / 2.18689567
a = np.diff(spikes['times'][np.where(np.diff(spikes['samples']) > (0.007 * 30_000))]) / (65600 / 30000)
ax.hist(a, bins=500)

#ax.hist(a, bins=20, range=[0, 5])
ax.set(ylabel='Count of spike times > 7 ms', xlabel='Kilosort batch', xlim=[0, 5])
sns.despine(trim=False)
plt.tight_layout()