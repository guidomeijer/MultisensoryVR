# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:59:25 2025

By Guido Meijer
"""

import spikeinterface.full as si
from os.path import join
from msvr_functions import paths
import matplotlib.pyplot as plt

# Settings
SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe01'

# Load in data
path_dict = paths()
probe_path = join(path_dict['local_data_path'], 'Subjects', f'{SUBJECT}', f'{DATE}', 'raw_ephys_data', f'{PROBE}')

# Load lfp
rec = si.load(join(probe_path, 'lfp_raw_binary'))

time_start = 3300
time_end = 3300 + 120
samples_start = int(time_start * rec.sampling_frequency)
samples_end = int(time_end * rec.sampling_frequency)

lfp_traces = rec.get_traces(start_frame=samples_start, end_frame=samples_end, 
                            channel_ids=['AP303', 'AP309', 'AP315'])




