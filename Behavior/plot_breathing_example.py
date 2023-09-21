# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:00:55 2023 by Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from msvr_functions import paths, figure_style


# Get paths
path_dict = paths()
data_path = path_dict['local_data_path']

# Load in data
breathing = np.load(join(data_path, '450508', '20230920', 'continuous.breathing.npy'))
timestamps = np.load(join(data_path, '450508', '20230920', 'continuous.times.npy'))

# Plot
colors, dpi = figure_style()
f, ax1 = plt.subplots(figsize=(1.75, 2), dpi=dpi)
ax1.plot(timestamps[:1000], breathing[:1000])

plt.tight_layout()
sns.despine(trim=True, left=True)
