#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:51:57 2025

@author: Guido
"""

import os
from os import path
import shutil
from run_pipeline import run_pipeline

# Settings
SERVER_PATH = '/mnt/imaging1/imaging1/guido/Subjects'
LOCAL_PATH = '/home/user/Data/guido/Subjects/'

for root, directory, files in os.walk(SERVER_PATH):
    if 'process_me.flag' in files:
        
        # Copy files to local SSD
        print(f'\nFound process_me.flag in {root}\nCopying files to SSD..')
        subject = path.split(path.split(root)[0])[-1]
        date = path.split(root)[-1]
        if not path.isdir(path.join(LOCAL_PATH, subject, date)):
            shutil.copytree(root, path.join(LOCAL_PATH, subject, date))
        
        # Run PowerPixels pipeline on local data
        run_pipeline()
        
        # Copy results to server
        print('\nCopying processed session back to server')
        shutil.copytree(path.join(LOCAL_PATH, subject, date),
                        path.join(SERVER_PATH, subject, f'{date}_processed'))
        
        # Delete original session
        shutil.rmtree(path.join(SERVER_PATH, subject, date))
        
        # Rename processed session to original name
        os.rename(path.join(SERVER_PATH, subject, f'{date}_processed'),
                  path.join(SERVER_PATH, subject, date))
        
        # Delete session from local SSD to make room for the next one
        shutil.rmtree(path.join(LOCAL_PATH, subject, date))
        
        
        
        
        
