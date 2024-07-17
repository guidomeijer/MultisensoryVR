# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:06:13 2023

@author: Neuropixel
"""

import os
from os.path import join, split, isfile, isdir
import shutil
from glob import glob
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--ephys", help="Ephys session", action='store_true')
parser.add_argument("-o", "--overwrite", help="Overwrite", action='store_true')
args = parser.parse_args()

if args.ephys:
    LOCAL_FOLDER = 'D:\\NeuropixelData'
else:
    LOCAL_FOLDER = 'K:\\Subjects'
REMOTE_FOLDER = 'U:\\imaging1\\guido\\Subjects'

# Search for transfer_me.flag
print('Looking for transfer_me.flag')
for root, directory, files in os.walk(LOCAL_FOLDER):
    if 'transfer_me.flag' in files:
        print(f'\n{root}\n')
        
        # Don't transfer session if it's already on the server
        server_folder = join(REMOTE_FOLDER, split(split(root)[0])[1], split(root)[1])
        if isdir(server_folder) and not args.overwrite:
            print(f'Session {root} already on server, skipping')
            os.remove(join(root, 'transfer_me.flag'))
            continue
        
        # Compress video
        h264_path = glob(join(root, 'raw_video_data', '*.h264'))
        for i, raw_video_path in enumerate(h264_path):  # in principle there should be only one
            mp4_path = raw_video_path[:-5] + '.mp4'
            print(raw_video_path)
            print(mp4_path)
            if isfile(mp4_path):
                continue
            subprocess.call(['ffmpeg', '-i', raw_video_path, '-vcodec', 'libx265', '-crf', '20',
                             '-n', mp4_path])
            if isfile(mp4_path) and isfile(raw_video_path):
                os.remove(raw_video_path)        
       
        # Copy data
        print(f'\nCopying session {root} to server')
        shutil.copytree(root, server_folder)
       
        # Delete transfer_me flags
        os.remove(join(root, 'transfer_me.flag'))
        os.remove(join(server_folder, 'transfer_me.flag'))
        print('Done!')
        
        