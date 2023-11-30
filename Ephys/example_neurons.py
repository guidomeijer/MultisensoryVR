# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:32:45 2023

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from brainbox.plot import peri_event_time_histogram
from msvr_functions import paths, figure_style, load_spikes

# Settings
SUBJECT = '450409'
DATE = '20231012'
PROBE = 'probe00'

# Get paths
path_dict = paths()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters = load_spikes(session_path, PROBE)
trials = pd.read_csv(join(path_dict['local_data_path'], 'Subjects', SUBJECT, DATE, 'trials.csv'))



spike_times = np.load(r'example_data\spike_times_example.npy')
ii_spike_times = np.load(r'example_data\ii_spike_times_example.npy')
epoch_labels = np.load(r'example_data\epoch_labels_example.npy')