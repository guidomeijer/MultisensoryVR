# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 13:24:04 2025

@author: leviv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import spikeinterface.full as se
from pathlib import Path
from scipy.signal import butter, filtfilt
from matplotlib.ticker import MultipleLocator
from collections import defaultdict



# Python code for bachelor internship


# Loading data from file

# Change path to your local directory containing the probe file
#SES_PATH = Path(r'V:\imaging1\guido\Subjects\466395\20241114')
#SES_PATH = Path(r'V:\imaging1\guido\Subjects\466396\20241101')   # good
SES_PATH = Path(r'V:\imaging1\guido\Subjects\466396\20241031')  # good
#SES_PATH = Path(r'V:\imaging1\guido\Subjects\462910\20240813')  # good


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
#traces = rec.get_traces(start_frame=rec.time_to_sample_index(last_trial + 300),
#                        end_frame=rec.time_to_sample_index(last_trial + 301))

# Get the channel information
channels_df = pd.read_csv(SES_PATH / 'probe01' / 'channels.brainLocation.csv')

# Load spectral power data
spec_power = np.load(SES_PATH / 'probe01' / '_iblqc_ephysSpectralDensityLF.power.npy')
# Load spectral frequency data
spec_freq = np.load(SES_PATH / 'probe01' / '_iblqc_ephysSpectralDensityLF.freqs.npy')
# Load local coordinates
coords = np.load(SES_PATH / 'probe01' / 'channels.localCoordinates.npy')



loc_brain = pd.read_csv(SES_PATH / 'probe01' / 'channels.brainLocation.csv')
raw_ind = np.load(SES_PATH / 'probe01' / 'channels.rawInd.npy')

# Throw out faulty neuropixel channels
spec_power = spec_power[:, raw_ind]

# Make a map for shank selection | id : lateral_um values
shank_lateral_map = {
    1: [0.0, 32.0],
    2: [250.0, 282.0],
    3: [500.0, 532.0],
    4: [750.0, 782.0]
}


def map_coords(data, coords):
    """
    Map a T x N data array to its spatial coordinates.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (T, N) where N = number of spatial points
    coords : np.ndarray
        Array of shape (N, 2) with x and z coordinates

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (T, N), with columns labeled as (x,z) tuples
    """
    data = np.asarray(data)
    coords = np.asarray(coords)
    
    if data.shape[1] != coords.shape[0]:
        raise ValueError("data.shape[1] must equal coords.shape[0]")
    
    # Create column labels as (x, z) tuples
    col_labels = [tuple(coord) for coord in coords]
    
    df = pd.DataFrame(data, columns=col_labels)
    return df

def split_prongs(df):
    """
    Split a DataFrame of shape (T, 384) into 8 prongs based on x coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned from map_coords_time, with columns labeled as (x,z)

    Returns
    -------
    list of pd.DataFrame
        8 DataFrames, one per x-value (prong), containing all T rows and the columns
        corresponding to that x-value.
    """
    # Hard-coded x-values
    x_values = [0.0, 32.0, 250.0, 282.0, 500.0, 532.0, 750.0, 782.0]
    
    prongs = []
    
    # Extract column coordinates
    coords = np.array(df.columns)  # shape (384, 2)
    x_cols = np.array([col[0] for col in df.columns]) 
    
    for x in x_values:
        mask = x_cols == x
        prong_df = df.iloc[:, mask]
        prongs.append(prong_df)
    
    return prongs

def sort_prong(prong):
    """
    Sort the columns of a prong DataFrame by z-coordinate (second element of the column tuple).

    Parameters
    ----------
    prong : pd.DataFrame
        DataFrame corresponding to a single x-coordinate prong, columns are (x,z) tuples

    Returns
    -------
    pd.DataFrame
        Prong DataFrame with columns sorted by z-coordinate (ascending)
    """
    # Extract z-values from column labels
    coords = np.array(prong.columns)  # shape (num_columns, 2)
    z_vals = np.array([col[1] for col in prong.columns])
    
    # Get sort order
    sort_idx = np.argsort(z_vals)
    
    # Reorder columns
    prong_sorted = prong.iloc[:, sort_idx]
    
    return prong_sorted

# Select the correct traces to load into memory, based on start time and time window and shank
def get_traces(rec, start_s, win_s, SF=SF):
    """
    Load a chunk of traces from a recording for a specific shank.
    
    Parameters
    ----------
    rec : SpikeInterface recording object
    start_s : float
        Start time in seconds
    win_s : time window of 
    SF : float
        Sampling frequency
    
    Returns
    -------
    traces_chunk : np.ndarray
        Array of shape (n_samples_chunk, n_channels_shank)
    """
    start_frame = rec.time_to_sample_index(start_s)
    end_frame = rec.time_to_sample_index(start_s + win_s)
    traces = rec.get_traces(start_frame=start_frame, end_frame=end_frame)
    
    return traces





def band_pass_filter(traces, SF=2500, lowcut=30.0, highcut=100.0, order=4):
    """
    Apply a band-pass Butterworth filter to LFP traces.

    Parameters
    ----------
    traces : np.ndarray
        Shape (n_samples, n_channels)
    SF : float
        Sampling frequency in Hz
    lowcut : float
        Lower cutoff frequency in Hz
    highcut : float
        Upper cutoff frequency in Hz
    order : int
        Order of the Butterworth filter

    Returns
    -------
    filtered_traces : np.ndarray
        Shape (n_samples, n_channels), filtered LFP
    """
    nyq = 0.5 * SF
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter along time axis (axis=0)
    filtered_traces = filtfilt(b, a, traces, axis=0)
    
    return filtered_traces


def plot_scroller(prongs, shank_id=1, SF=2500, n_ch=8, spacing=100, win_ms=200, step_ms=10):
    """
    Scrollable plot for 2 prongs per shank with keyboard support.
    Redraws full window each scroll to avoid data mismatch issues.
    """
    assert 1 <= shank_id <= 4
    prong_indices = [(shank_id-1)*2, (shank_id-1)*2 + 1]
    selected_prongs = [prongs[i] for i in prong_indices]

    n_samples = selected_prongs[0].shape[0]
    win_samples = int(win_ms / 1000 * SF)
    step_samples = int(step_ms / 1000 * SF)

    # Time axis function
    def get_time(start):
        return np.arange(start, start+win_samples)/SF*1000  # ms

    start_idx = 0

    # --- Figure setup ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    plt.subplots_adjust(bottom=0.25)

    def redraw(start):
        x = get_time(start)
        for ax_idx, prong in enumerate(selected_prongs):
            ax = axes[ax_idx]
            ax.clear()
            ch_data = prong.iloc[start:start+win_samples, :n_ch].to_numpy()
            for i in range(n_ch):
                y = ch_data[:, i] + i*spacing
                ax.plot(x, y, color='k', lw=0.75)
                ax.text(x[0]-0.05, i*spacing, f"Ch {i}", va='center', ha='right', fontsize=10)
            ax.set_title(f'Prong {prong_indices[ax_idx]+1}')
            ax.set_xlim(x[0], x[-1])
            
            margin = 2.5 * spacing
            ax.set_ylim(-spacing - margin, n_ch*spacing + spacing + margin)
            ax.set_yticks([])
            ax.set_xticks(np.arange(x[0], x[-1]+1, 5))
        axes[0].set_ylabel('Channels stacked')       
        fig.canvas.draw_idle()

    redraw(start_idx)

    # --- Slider ---
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.12, 0.1, 0.75, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Start sample', 0, n_samples-win_samples, valinit=start_idx, valstep=1)

    def update(val):
        nonlocal start_idx
        start_idx = int(slider.val)
        redraw(start_idx)

    slider.on_changed(update)

    # --- Keyboard support ---
    def on_key(event):
        nonlocal start_idx
        if event.key == 'd':
            start_idx = min(start_idx + step_samples, n_samples-win_samples)
        elif event.key == 'a':
            start_idx = max(start_idx - step_samples, 0)
        slider.set_val(start_idx)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()



    
def spec_heatmap(spec_power, spec_freq, coords, band=(50,100)):
    """
    Group `array` values by unique x positions (coords[:,0]), 
    and sort within each group by z position (coords[:,1]).
    
    Parameters
    ----------
    array : np.ndarray
        1D array of data values (length N)
    coords : np.ndarray
        2D array of shape (N, 2), first column = x, second column = z

    Returns
    -------
    np.ndarray
        2D array of shape (num_points_per_group, num_groups)
        Columns correspond to unique x-values, rows sorted by z.
    np.ndarray
        The sorted unique x-values (to label columns in the heatmap)
    """
    
    f_lo, f_hi = band
    freq_mask = (spec_freq >= f_lo) & (spec_freq <= f_hi)
    
    # Mean power in band per channel
    band_power = spec_power[freq_mask, :].mean(axis=0)
    band_power_db = 10 * np.log10(band_power + 1e-20)
    
    array = np.asarray(band_power_db)
    coords = np.asarray(coords)
    
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be shape (N,2)")
    
    x_unique = np.sort(np.unique(coords[:,0]))   # 8 unique x-values
    num_groups = len(x_unique)
    
    # Determine max number of points per group (if unequal, will pad with np.nan)
    max_len = max(np.sum(coords[:,0] == x) for x in x_unique)
    
    # Initialize 2D array with NaNs
    heatmap_array = np.full((max_len, num_groups), np.nan)
    
    for i, x_val in enumerate(x_unique):
        mask = coords[:,0] == x_val
        z_vals = coords[mask,1]
        data_vals = array[mask]
        
        # Sort by z
        order = np.argsort(z_vals)
        heatmap_array[:len(data_vals), i] = data_vals[order]
    
    return heatmap_array, x_unique


def run_scroller():
    #Code for running scroller, change parameters here
    
        
    # Parameters
    shank_id = 4
    start_s = last_trial + 600 # in seconds
    win_s = 20 # in seconds
    win_scroller = 1000 # in ms
    lowcut = 5
    highcut = 500
    order = 1
    n_ch = 48
    spacing = 35
    scroll_speed = win_scroller/5
        
    
    traces = get_traces(rec, start_s, win_s)
    filtered_traces = band_pass_filter(traces, SF, lowcut, highcut, order)
    filtered_traces = filtered_traces / 5
    traces_df = map_coords(filtered_traces, coords)
    prongs = split_prongs(traces_df)
    sorted_prongs = [sort_prong(p) for p in prongs]
    
    plot_scroller(sorted_prongs, shank_id, SF, n_ch, spacing, win_scroller, scroll_speed)


# Code for running heatmap plotter, change parameters here
def run_heatmap():
    heatmap, prongs = spec_heatmap(spec_power, spec_freq, coords, band=(150,250))
    
    plt.figure()
    plt.imshow(heatmap, aspect='auto', origin='lower', cmap='hot')
    plt.xticks(ticks=np.arange(len(prongs)), labels=prongs)
    plt.xlabel('X position')
    plt.ylabel('Depth (Z)')
    plt.colorbar(label='Band power (50-100 Hz)')
    plt.show()
    



# Toggle functions to run scroller vs heatmap respectively
# To change parameters, edit run_scroller and run_heatmap above
    
run_scroller()
#run_heatmap()