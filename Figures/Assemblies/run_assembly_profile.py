# -*- coding: utf-8 -*-
"""
Author: Guido Meijer
Date: 10/03/2026
"""
# %%

import numpy as np
import matplotlib.pyplot as plt
from msvr_functions import paths

path_dict = paths()

# Get data files
activation_files = (path_dict['google_drive_data_path'] / 'RippleAssemblies').rglob('*mean_activation.npy')
time_files = (path_dict['google_drive_data_path'] / 'RippleAssemblies').rglob('*time.npy')
time_ax = np.load(list(time_files)[0])

# Load in data per region
assembly_act = {'CA1': [], 'LEC': [], 'AUD': [], 'VIS': [], 'TEa': [], 'PERI': []}
for i, file in enumerate(activation_files):
    this_region = file.stem.split('_')[3]
    this_act = np.load(file)
    assembly_act[this_region].append(this_act)

# Loop over regions
for r, region in enumerate(assembly_act.keys()):
    assembly_act[region] = np.vstack(assembly_act[region])

plt.imshow(assembly_act['CA1']); plt.show()