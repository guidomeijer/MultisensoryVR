# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 11:54:55 2026

By Guido Meijer
"""

import json
from ibllib.pipes import histology
from pathlib import Path
from iblatlas import atlas
ba = atlas.AllenAtlas(25)

PATH = Path(r'D:\Histology\478154')

for file in PATH.glob('*.csv'):
    print(f'Processing {file.stem}')
    xyz_track = histology.load_track_csv(file, brain_atlas=ba)
    insertion = atlas.Insertion.from_track(xyz_track, brain_atlas=ba)
    xyz_dict = {'xyz_picks': [list(insertion.tip * 1e6), list(insertion.entry * 1e6)]}
    with open(file.parent / (file.stem + '.json'), 'w') as fp:
        json.dump(xyz_dict, fp)
    

