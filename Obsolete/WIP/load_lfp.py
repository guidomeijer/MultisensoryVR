# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 13:45:05 2025

By Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.full as se
from pathlib import Path

SES_PATH = Path(r'D:\MultisensoryVR\Subjects\466395\20241115')
RAW_FILE = SES_PATH / 'probe01' / 'lfp_raw_binary' / 'traces_cached_seg0.raw'
SF = 2500  # sampling frequency

# Lazy reading in of the recording
rec = se.read_binary(RAW_FILE, sampling_frequency=SF, dtype='int16', num_channels=384, t_starts=[0])
print(rec)

# Get timestamps
timestamps = rec.get_times()

# Load in trials
trials_df = pd.read_csv(SES_PATH / 'trials.csv')
last_trial = trials_df['exitEnvTime'].values[-1]

# Extracting the actual traces, this goes straight into the RAM so only load in chunks
# start_frame is the first sample of the chunk and end_frame the last one (this is samples not time!)
# Let's read in 10 seconds starting from 5 minutes after the last trial
traces = rec.get_traces(start_frame=rec.time_to_sample_index(last_trial + 60),
                        end_frame=rec.time_to_sample_index(last_trial + 70))

# Get the channel information
channels_df = pd.read_csv(SES_PATH / 'probe01' / 'channels.brainLocation.csv')

# Find the deepest channel
deepest_ch = channels_df['z'].argmin()

# Now let's plot the deepest channel
f, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
ax1.plot(traces[:, deepest_ch])
plt.tight_layout()
