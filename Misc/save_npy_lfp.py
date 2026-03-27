# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 05/03/2026
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import spikeinterface.full as si
from probeinterface import get_probe, write_probeinterface

# Settings
SES_PATH = Path(r'E:\Test')
SAVE_PATH = Path(r'E:\Test')
TIME_FROM_END = (20 * 60) # s

# Load in LFP data
rec = si.read_binary(SES_PATH / 'lfp_raw_binary' / 'traces_cached_seg0.raw',
                     sampling_frequency=2500, dtype='int16', num_channels=384)

# Import probe interface and set to Neuropixel 2.0 four-shank
probe = get_probe('imec', 'NP2014')
write_probeinterface(SAVE_PATH / 'probe_config.json', probe)

# Export LFP data as NPY
start_frame = rec.time_to_sample_index(rec.get_duration() - TIME_FROM_END)
lfp_traces = rec.get_traces(start_frame=start_frame, end_frame=rec.get_total_samples())
np.save(SAVE_PATH / 'lfp_traces.npy', lfp_traces)
