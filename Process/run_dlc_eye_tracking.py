import os
from pathlib import Path
import pandas as pd
import numpy as np
import deeplabcut
import subprocess
import shutil
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from ellipse import LsqEllipse
lsqe = LsqEllipse()

DATA_PATH = Path('/vol/battaglialab/imaging1/Angela/2_Experiments/2021-0052-018_Neural correlates of memories')
DLC_FIND_EYE = Path('/vol/battaglialab/imaging1/guido/DLC/find-eye-guido-2023-11-10/config.yaml')
DLC_EYE_TRACK = Path('/vol/battaglialab/imaging1/guido/DLC/pupil-tracking-guido-2023-11-13/config.yaml')
EYE_WIDTH_PX = 80
EYE_HEIGHT_PX = 70
MIN_PROB = 0.7  # minimum probablitiy of tracked points to contribute to pupil fitting
MIN_POINTS = 5  # minimum number of points to fit pupil ellipse
MAX_WH_RATIO = 1.5  # maximum ratio between width and height to prevent bad fits
EYE_FLAG = 'eyetrack_me.flag'

# %% Functions
def fit_ellipse(i, eye_dlc):
    """
    Fits an ellipse to eye tracking data for a given frame or index.
    Parameters
    ----------
    i : int or hashable
        The index or frame number to extract from the eye_dlc DataFrame.
    eye_dlc : pandas.DataFrame
        A multi-indexed DataFrame containing eye tracking data with columns for 'x', 'y', and 'likelihood' 
        at the second level.
    Returns
    -------
    center_x : float or np.nan
        The x-coordinate of the ellipse center, or np.nan if fitting fails.
    center_y : float or np.nan
        The y-coordinate of the ellipse center, or np.nan if fitting fails.
    width : float or np.nan
        The width (major axis) of the fitted ellipse, or np.nan if fitting fails.
    height : float or np.nan
        The height (minor axis) of the fitted ellipse, or np.nan if fitting fails.
    phi : float or np.nan
        The rotation angle of the ellipse in radians, or np.nan if fitting fails.
    Notes
    -----
    The function filters points based on a minimum likelihood threshold (MIN_PROB) and requires at least 
    MIN_POINTS to attempt fitting. If the width-to-height ratio exceeds MAX_WH_RATIO, the fit is considered invalid.
    """

    
    x = eye_dlc.xs('x', level='coords', axis=1).loc[i].values
    y = eye_dlc.xs('y', level='coords', axis=1).loc[i].values
    xy_prob = eye_dlc.xs('likelihood', level='coords', axis=1).loc[i].values
    if np.sum(xy_prob > MIN_PROB) >= MIN_POINTS:
        x = x[xy_prob > MIN_PROB]
        y = y[xy_prob > MIN_PROB]
        lsqe.fit(np.stack((x, y)).T)
        try:
            center, width, height, phi = lsqe.as_parameters()
            center_x, center_y = center[0], center[1]
        except Exception:
            center_x, center_y, width, height, phi = np.nan, np.nan, np.nan, np.nan, np.nan
        if np.abs(width/height) > MAX_WH_RATIO:
            center_x, center_y, width, height, phi = np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        center_x, center_y, width, height, phi = np.nan, np.nan, np.nan, np.nan, np.nan
    
    return center_x, center_y, width*2, height*2, phi


def smooth_pupil(signal, window=61, order=3, interp_kind='cubic'):
    """Run savitzy-golay filter on signal, interpolate through nan points.

    Parameters
    ----------
    signal : np.ndarray
        original noisy signal of shape (t,), may contain nans
    window : int
        window of polynomial fit for savitzy-golay filter
    order : int
        order of polynomial for savitzy-golay filter
    interp_kind : str
        type of interpolation for nans, e.g. 'linear', 'quadratic', 'cubic'
    Returns
    -------
    np.array
        smoothed, interpolated signal for each time point, shape (t,)

    """

    signal_noisy_w_nans = np.copy(signal)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    good_idxs = np.where(~np.isnan(signal_noisy_w_nans))[0]
    # perform savitzky-golay filtering on non-nan points
    signal_smooth_nonans = non_uniform_savgol(
        timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
    signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
    signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
    # interpolate nan points
    interpolater = interp1d(
        timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')

    signal = interpolater(timestamps)

    return signal


def non_uniform_savgol(x, y, window, polynom):
    """Applies a Savitzky-Golay filter to y with non-uniform spacing as defined in x.
    This is based on
    https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do
    https://dsp.stackexchange.com/a/64313
    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size
    Returns
    -------
    np.array
        The smoothed y values
    """

    half_window = window // 2
    poly_order = polynom + 1

    y_smoothed = np.full(len(y), np.nan)

    # Precompute powers for efficiency
    def vandermonde(x, center, order):
        t = x - center
        return np.vstack([t ** k for k in range(order)]).T

    # Store coefficients for border interpolation
    first_coeffs = None
    last_coeffs = None

    for i in range(half_window, len(x) - half_window):
        x_win = x[i - half_window:i + half_window + 1]
        y_win = y[i - half_window:i + half_window + 1]
        A = vandermonde(x_win, x[i], poly_order)
        # Solve least squares for polynomial coefficients
        coeffs, *_ = np.linalg.lstsq(A, y_win, rcond=None)
        y_smoothed[i] = coeffs[0]
        if i == half_window:
            first_coeffs = coeffs
            first_center = x[half_window]
        elif i == len(x) - half_window - 1:
            last_coeffs = coeffs
            last_center = x[-half_window - 1]

    # Interpolate the result at the left border
    if first_coeffs is not None:
        for i in range(half_window):
            t = x[i] - first_center
            y_smoothed[i] = np.polyval(first_coeffs[::-1], t)

    # Interpolate the result at the right border
    if last_coeffs is not None:
        for i in range(len(x) - half_window, len(x)):
            t = x[i] - last_center
            y_smoothed[i] = np.polyval(last_coeffs[::-1], t)

    return y_smoothed


# %% Main script to process eye tracking data
for root, directory, files in os.walk(DATA_PATH / 'Raw_Data_READONLY'):
    if EYE_FLAG in files:
        root = Path(root)
        print(f'\nFound {EYE_FLAG} in {root}')
        
        # Get path to video file
        video_dir = [i for i in directory if i[-9:] == '_picamera']
        if len(video_dir) != 1:
            print('More or fewer than 1 video in {root}')
            continue  
        else:
            video_dir = video_dir[0]
        video_path = list((Path(root) / video_dir).rglob('*.mp4'))
        if len(video_path) != 1:
            print('More or fewer than 1 video in {root}')
            continue
        else:
            video_path = video_path[0]
            
        # Extract a single frame from the video
        subprocess.call(['ffmpeg', '-y', '-ss', '00:05:00', '-i', video_path,
                         '-frames:v', '1', '-q:v', '2',
                         video_path.parent / 'single_frame.jpg'])
        
        # Run DLC on single frame to determine the position of the eye
        print('\nRun DLC on single frame to find eye position')
        deeplabcut.analyze_time_lapse_frames(DLC_FIND_EYE, video_path.parent,
                                             frametype='.jpg', save_as_csv=True)
        
        # Get position of the eye
        dlc_file = list(video_path.parent.rglob('*find-eye*.csv'))[0]
        dlc_output = pd.read_csv(dlc_file)
        eye_x = int(float(dlc_output.iloc[3, 1]))
        eye_y = int(float(dlc_output.iloc[3, 2]))

        # Crop out the eye into new video
        eye_path = video_path.with_name(f'{video_path.stem}_eyecrop{video_path.suffix}')
        if not eye_path.is_file():
            print('\nCrop eye out of video')
            subprocess.call([
                'ffmpeg', '-i', video_path, '-vf',
                f'crop={EYE_WIDTH_PX}:{EYE_HEIGHT_PX}:{int(eye_x-EYE_WIDTH_PX/2)}:{int(eye_y-EYE_HEIGHT_PX/2)}',
                '-c:v', 'libx265', '-crf', '0', '-c:a', 'copy',
                '-y', eye_path])
        
        # Apply curve to increase low to mid level brightness
        eye_adjust_path = eye_path.with_name(f'{eye_path.stem}_adjusted{eye_path.suffix}')
        if not eye_adjust_path.is_file():
            subprocess.call([
                'ffmpeg', '-i', eye_path, '-vf', "curves=all='0/0 0.2/0.6 1/1',format=yuv420p",
                '-c:a', 'copy', '-y', eye_adjust_path])

        # Track pupil using pre-trained model
        print('\nStart eye tracking')
        deeplabcut.analyze_videos(DLC_EYE_TRACK, eye_adjust_path, save_as_csv=True)
        
        # Create labelled video
        deeplabcut.create_labeled_video(DLC_EYE_TRACK, [eye_adjust_path], save_frames=False)
        label_local_path = list(video_path.parent.rglob('*labeled.mp4'))[0]
        
        # Filter traces
        deeplabcut.filterpredictions(DLC_EYE_TRACK, [eye_adjust_path])
        
        # Get pupil by fitting elipse using least squares method
        if not (root / 'pupil.csv').is_file():
            dlc_out = list(video_path.parent.rglob('*pupil-tracking*_filtered.csv'))[0]
            eye_dlc = pd.read_csv(dlc_out, header=[0, 1, 2], index_col=0)
            eye_df = pd.DataFrame()
            print('\nFitting ellipse to tracked points')

            results = Parallel(n_jobs=-1)(delayed(fit_ellipse)(i, eye_dlc)
                                         for i in range(eye_dlc.shape[0]))
            eye_df = pd.DataFrame(data={'center_x': [res[0] for res in results],
                                        'center_y': [res[1] for res in results],
                                        'width': [res[2] for res in results],
                                        'height': [res[3] for res in results],
                                        'phi': [res[4] for res in results]})

            # Smooth pupil diameter
            if np.sum(np.isnan(eye_df['width'])) / eye_df.shape[0] > 0.75:
                print('Could not find the pupil by ellipse fitting')
            else:
                print('\nSmoothing pupil traces')
                eye_df['width_smooth'] = smooth_pupil(eye_df['width'])
                eye_df['height_smooth'] = smooth_pupil(eye_df['height'])
                
                # Save pupil tracking to server
                eye_df.to_csv(root / 'pupil.csv', index=False)
                print('Pupil traces saved to server')
        
        # Delete eyetrack_me.flag from server
        os.remove(root / 'eyetrack_me.flag')
        print('\nDone! Deleted eyetrack_me.flag\n')
        
