# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:37:38 2025 by Guido Meijer
"""

import numpy as np
import pandas as pd
from os import path
from msvr_functions import load_objects, load_neural_data
from brainbox.population.decode import get_spike_counts_in_bins

SUBJECT = '459601'
DATE = '20240411'
PROBE = 'probe00'
REGION = 'TEa'
SES_PATH = r'C:\Users\guido\Data\MultisensoryVR\Subjects\459601\20240411'

# Load in object entry times
obj_df = load_objects(SUBJECT, DATE)

# Load in neural data
spikes, clusters, channels = load_neural_data(SES_PATH, PROBE)

# Get neurons from this brain region
region_neurons = clusters['cluster_id'][clusters['region'] == REGION]

# Get spike counts per trial 
get_spike_counts_in_bins
