import os
from os.path import join, split, isfile, isdir
import pandas as pd
from glob import glob
import numpy as np
import deeplabcut
import subprocess
import shutil
from pipeline_utils import smooth_pupil
from ellipse import LsqEllipse
lsqe = LsqEllipse()

SERVER_PATH = '/mnt/imaging1/guido/Subjects'
LOCAL_PATH = '/home/user/Data/guido/Subjects'
DLC_FIND_EYE = '/home/user/DLC/find-eye-guido-2023-11-10/config.yaml'
DLC_EYE_TRACK = '/home/user/DLC/pupil-tracking-guido-2023-11-13/config.yaml'
EYE_WIDTH_PX = 80
EYE_HEIGHT_PX = 70
MIN_PROB = 0.99  # minimum probablitiy of tracked points to contribute to pupil fitting
MIN_POINTS = 5  # minimum number of points to fit pupil ellipse
MAX_WH_RATIO = 1.5  # maximum ratio between width and height to prevent bad fits

for root, directory, files in os.walk(SERVER_PATH):
    if 'eyetrack_me.flag' in files:
        print(f'\nFound eyetrack_me.flag in {root}')
        
        h264_server_path = glob(join(root, 'raw_video_data', '*.h264'))
        if len(h264_server_path) == 0:
            print(f'No video found in {join(root, "raw_video_data")}')
            os.remove(join(root, 'eyetrack_me.flag'))
            continue
        elif len(h264_server_path) > 1:
            print(f'Multiple videos found in {join(root, "raw_video_data")}')
            continue
        elif len(h264_server_path) == 1:
            h264_server_path = h264_server_path[0]
        
        # Copy to local disk for processing
        subject = split(split(split(split(h264_server_path)[0])[0])[0])[1]
        date = split(split(split(h264_server_path)[0])[0])[1]
        if not isdir(join(LOCAL_PATH, subject)):
            os.mkdir(join(LOCAL_PATH, subject))
        if not isdir(join(LOCAL_PATH, subject, date)):
            os.mkdir(join(LOCAL_PATH, subject, date))
        local_folder_path = join(LOCAL_PATH, subject, date)
        h264_local_path = join(local_folder_path, split(h264_server_path)[1])
        if not isfile(join(local_folder_path, split(h264_server_path)[1])):
            print('\nCopying video to local disk for processing')
            shutil.copy(h264_server_path, h264_local_path)
        
        # Convert video from .h264 to .mp4
        print('\nConvert video to mp4')
        mp4_local_path = join(local_folder_path, split(h264_server_path)[1][:-5] + '.mp4')
        subprocess.call(['ffmpeg', '-i', h264_local_path, '-codec', 'copy', '-n', mp4_local_path])
        
        # Extract a single frame from the video
        subprocess.call(['ffmpeg', '-ss', '00:10:00', '-i', mp4_local_path,
                         '-frames:v', '1', '-q:v', '2',
                         join(local_folder_path, 'single_frame.jpg')])
        
        # Run DLC on single frame to determine the position of the eye
        print('\nRun DLC on single frame to find eye position')
        deeplabcut.analyze_time_lapse_frames(DLC_FIND_EYE, local_folder_path,
                                             frametype='.jpg', save_as_csv=True)
        
        # Get position of the eye
        dlc_file = glob(join(local_folder_path, '*find-eye*.csv'))
        dlc_output = pd.read_csv(dlc_file[0], header=[1, 2], index_col=0)
        eye_x = int(dlc_output.xs('x', level=1, axis=1).values[0][0])
        eye_y = int(dlc_output.xs('y', level=1, axis=1).values[0][0])

        # Crop out the eye into new video
        eye_local_path = mp4_local_path[:-4] + '_eyecrop.mp4'
        if not isfile(eye_local_path):
            print('\nCrop eye out of video')
            subprocess.call([
                'ffmpeg', '-i', mp4_local_path, '-vf',
                f'crop={EYE_WIDTH_PX}:{EYE_HEIGHT_PX}:{int(eye_x-EYE_WIDTH_PX/2)}:{int(eye_y-EYE_HEIGHT_PX/2)}',
                '-c:v', 'libx264', '-crf', '0', '-c:a', 'copy',
                '-n', eye_local_path])
        
        # Track pupil using pre-trained model
        print('\nStart eye tracking')
        deeplabcut.analyze_videos(DLC_EYE_TRACK, eye_local_path, save_as_csv=True)
        
        # Create labelled video
        deeplabcut.create_labeled_video(DLC_EYE_TRACK, [eye_local_path], save_frames=False)
        label_local_path = glob(join(local_folder_path, '*labeled.mp4'))[0]
        
        # Filter traces
        deeplabcut.filterpredictions(DLC_EYE_TRACK, [eye_local_path])
        
        # Get pupil by fitting elipse using least squares method
        if not isfile(join(local_folder_path, 'pupil.csv')):
            dlc_out = glob(join(local_folder_path, '*pupil-tracking*_filtered.csv'))[0]
            eye_dlc = pd.read_csv(dlc_out, header=[1, 2], index_col=0)
            eye_df = pd.DataFrame()
            print('\nFitting ellipse to tracked points')
            for i in eye_dlc.index.values:
                if np.mod(i, 5000) == 0:
                    print(f'Video frame {i} of {eye_dlc.shape[0]}')
                x = eye_dlc.xs('x', level=1, axis=1).loc[i].values
                y = eye_dlc.xs('y', level=1, axis=1).loc[i].values
                xy_prob = eye_dlc.xs('likelihood', level=1, axis=1).loc[i].values
                if np.sum(xy_prob > MIN_PROB) >= MIN_POINTS:
                    x = x[xy_prob > MIN_PROB]
                    y = y[xy_prob > MIN_PROB]
                    data = np.stack((x, y)).T
                    lsqe.fit(np.stack((x, y)).T)
                    center, width, height, phi = lsqe.as_parameters()
                    center_x, center_y = center[0], center[1]
                    if np.abs(width/height) > MAX_WH_RATIO:
                        center_x, center_y, width, height, phi = np.nan, np.nan, np.nan, np.nan, np.nan
                else:
                    center_x, center_y, width, height, phi = np.nan, np.nan, np.nan, np.nan, np.nan
                eye_df = pd.concat((eye_df, pd.DataFrame(index=[eye_df.shape[0]+1], data={
                    'center_x': center[0], 'center_y': center[1],
                    'width': width*2, 'height': height*2, 'phi': phi})))
              
            # Smooth pupil diameter
            print('\nSmoothing pupil traces')
            eye_df['width_smooth'] = smooth_pupil(eye_df['width'])
            eye_df['height_smooth'] = smooth_pupil(eye_df['height'])
            
            # Save pupil tracking to disk
            eye_df.to_csv(join(local_folder_path, 'pupil.csv'), index=False)
        
        # Compress video
        print('\nCompressing video')
        compr_local_path = mp4_local_path[:-4] + '_compressed.mp4'
        subprocess.call(['ffmpeg', '-i', mp4_local_path, '-vcodec', 'libx265', '-crf', '20', '-n',
                         compr_local_path])
        
        # Copy results to server
        print('\nCopying results to server')
        shutil.copy(eye_local_path, join(root, 'raw_video_data', split(eye_local_path)[1]))
        shutil.copy(compr_local_path, join(root, 'raw_video_data', split(compr_local_path)[1]))
        shutil.copy(label_local_path,
                    join(root, 'raw_video_data', split(mp4_local_path)[1][:-4] + '_labeled.mp4'))
        shutil.copy(join(local_folder_path, 'pupil.csv'), join(root, 'pupil.csv'))
        
        # Delete original uncompressed video from server
        if isfile(join(root, 'raw_video_data', split(compr_local_path)[1])):
            os.remove(h264_server_path)
                    
        # Create delete_me.flag to flag for future deletion
        with open(join(local_folder_path, 'delete_me.flag'), 'w') as fp:
            pass
        
        # Delete eyetrack_me.flag
        os.remove(join(root, 'eyetrack_me.flag'))
        print('\nDone! Deleted eyetrack_me.flag\n')