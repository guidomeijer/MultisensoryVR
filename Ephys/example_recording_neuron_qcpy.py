# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:32:45 2023

By Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from os.path import join, isfile
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from brainbox.plot import peri_event_time_histogram
from msvr_functions import (load_subjects, paths, peri_event_trace, figure_style,
                            peri_multiple_events_time_histogram, load_spikes)

# Settings
SUBJECT = '450409'
DATE = '20231012'
PROBE = 'probe00'
T_BEFORE = 0.5
T_AFTER = 1

# Get paths
path_dict = paths()

# Load in data
session_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}')
spikes, clusters = load_spikes(session_path, PROBE, only_good=False)

print(f'Total of {clusters["ks_label"].shape[0]} neurons')
print(f'{np.sum(clusters["ks_label"] == "good")} Kilosort good')
print(f'{np.sum(clusters["bc_label"] == "GOOD")} Bombcell good')
print(f'{np.sum(clusters["manual_label"] == "good")} manually annotated good')