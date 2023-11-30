# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:43:03 2023

By Guido Meijer
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import butter, filtfilt
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from brainbox import singlecell
from scipy.signal import gaussian, convolve
import json
import shutil
import datetime
from os.path import join, realpath, dirname, isfile, split, isdir


def paths(sync=True, full_sync=False, force_sync=False):
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input

    This function also runs the synchronization between the server and the local data folder 
    once a day

    Input
    ------------------------
    sync : boolean
        When True data from the server will be synced with the local disk
    full_sync : boolean 
        When True also the raw data will be copied to the local drive
    force_sync : boolean    
        When True synchronization will be done regardless of how long ago the last sync was    

    Output
    ------------------------
    path_dict : dictionary
        Dict with the paths
    """

    # Get the paths
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        path_dict = dict()
        path_dict['fig_path'] = input('Path folder to save figures: ')
        path_dict['server_path'] = input('Path folder to server: ')
        path_dict['local_data_path'] = input('Path folder to local data: ')
        path_dict['save_path'] = join(dirname(realpath(__file__)), 'Data')
        path_dict['repo_path'] = dirname(realpath(__file__))
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(path_dict, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        path_dict = json.load(json_file)

    # Synchronize data from the server with the local data folder

    # Create Subjects folder if it doesn't exist
    if not isdir(join(path_dict['local_data_path'], 'Subjects')):
        os.mkdir(join(path_dict['local_data_path'], 'Subjects'))

    # Read in the time of last sync
    if isfile(join(path_dict['local_data_path'], 'sync_timestamp.txt')):
        f = open(join(path_dict['local_data_path'], 'sync_timestamp.txt'), 'r')
        sync_time = datetime.datetime.strptime(f.read(), '%Y%m%d_%H%M')
        f.close()
    else:
        # If never been synched set date to yesterday so that it runs the first time
        sync_time = datetime.datetime.now() - datetime.timedelta(hours=24)

    # Synchronize server with local data once a day
    if sync:
        if ((datetime.datetime.now() - sync_time).total_seconds() > 12*60*60) | force_sync:
            print('Synchronizing data from server with local data folder')

            # Copy data from server to local folder
            subjects = os.listdir(join(path_dict['server_path'], 'Subjects'))
            for i, subject in enumerate(subjects):
                if not isdir(join(path_dict['local_data_path'], 'Subjects', subject)):
                    os.mkdir(join(path_dict['local_data_path'], 'Subjects', subject))
                sessions = os.listdir(join(path_dict['server_path'], 'Subjects', subject))
                for j, session in enumerate(sessions):
                    files = [f for f in os.listdir(join(path_dict['server_path'], 'Subjects', subject, session))
                             if (isfile(join(path_dict['server_path'], 'Subjects', subject, session, f))
                                 & (f[-4:] != 'flag'))]
                    if len(files) == 0:
                        continue
                    if not isdir(join(path_dict['local_data_path'], 'Subjects', subject, session)):
                        os.mkdir(join(path_dict['local_data_path'], 'Subjects', subject, session))
                    if not isfile(join(path_dict['local_data_path'], 'Subjects', subject, session, files[0])):
                        print(
                            f'Copying files {join(path_dict["server_path"], "Subjects", subject, session)}')
                        for f, file in enumerate(files):
                            if not isfile(join(path_dict['local_data_path'], 'Subjects', subject, session, file)):
                                shutil.copyfile(join(path_dict['server_path'], 'Subjects', subject, session, file),
                                                join(path_dict['local_data_path'], 'Subjects', subject, session, file))
                        if ((isdir(join(path_dict['server_path'], 'Subjects', subject, session, 'probe00')))
                                & (~isdir(join(path_dict['local_data_path'], 'Subjects', subject, session, 'probe00')))):
                            shutil.copytree(join(path_dict['server_path'], 'Subjects', subject, session, 'probe00'),
                                            join(path_dict['local_data_path'], 'Subjects', subject, session, 'probe00'))
                        if ((isdir(join(path_dict['server_path'], 'Subjects', subject, session, 'probe01')))
                                & (~isdir(join(path_dict['local_data_path'], 'Subjects', subject, session, 'probe01')))):
                            shutil.copytree(join(path_dict['server_path'], 'Subjects', subject, session, 'probe01'),
                                            join(path_dict['local_data_path'], 'Subjects', subject, session, 'probe01'))
                    if (not isdir(join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_video_data'))) & full_sync:
                        print(
                            f'Copying raw video data {join(path_dict["server_path"], "Subjects", subject, session)}')
                        shutil.copytree(join(path_dict['server_path'], 'Subjects', subject, session, 'raw_video_data'),
                                        join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_video_data'))
                    if isdir(join(path_dict['server_path'], 'Subjects', subject, session, 'raw_ephys_data')) & full_sync:
                        if not isdir(join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_ephys_data')):
                            print(
                                f'Copying raw ephys data {join(path_dict["server_path"], "Subjects", subject, session)}')
                            shutil.copytree(join(path_dict['server_path'], 'Subjects', subject, session, 'raw_ephys_data'),
                                            join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_ephys_data'))
                    if (not isdir(join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_behavior_data'))) & full_sync:
                        print(
                            f'Copying raw behavior data {join(path_dict["server_path"], "Subjects", subject, session)}')
                        shutil.copytree(join(path_dict['server_path'], 'Subjects', subject, session, 'raw_behavior_data'),
                                        join(path_dict['local_data_path'], 'Subjects', subject, session, 'raw_behavior_data'))

            # Update synchronization timestamp
            with open(join(path_dict['local_data_path'], 'sync_timestamp.txt'), 'w') as f:
                f.write(datetime.datetime.now().strftime('%Y%m%d_%H%M'))
                f.close()
            print('Done')

    return path_dict


def figure_style():
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 7,
                "figure.titlesize": 7,
                "axes.titlesize": 7,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                'legend.fontsize': 7,
                'legend.title_fontsize': 7,
                'legend.frameon': False,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    colors = {
        'obj1': sns.color_palette('Set2')[0],
        'obj2': sns.color_palette('Set2')[1],
        'obj3': sns.color_palette('Set2')[2],
        'goal': matplotlib.colors.to_rgb('mediumseagreen'),
        'no-goal': matplotlib.colors.to_rgb('tomato'),
        'control': matplotlib.colors.to_rgb('gray')}

    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 10
    return colors, dpi


def load_subjects():
    path_dict = paths(sync=False)
    subjects = pd.read_csv(join(path_dict['repo_path'], 'subjects.csv'),
                           delimiter=';|,',
                           engine='python')
    subjects['SubjectID'] = subjects['SubjectID'].astype(str)
    subjects['DateFinalTask'] = subjects['DateFinalTask'].astype(str)
    subjects['DateFinalTask'] = [datetime.datetime.strptime(i, '%Y%m%d').date() for i
                                 in subjects['DateFinalTask']]
    return subjects


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y


def bin_signal(timestamps, signal, bin_edges):
    bin_indices = np.digitize(timestamps[(timestamps >= bin_edges[0]) & (timestamps <= bin_edges[-1])],
                              bins=bin_edges, right=False) - 1
    bin_sums = np.bincount(
        bin_indices, weights=signal[(timestamps >= bin_edges[0]) & (timestamps <= bin_edges[-1])])
    bin_means = np.divide(bin_sums, np.bincount(bin_indices), out=np.zeros_like(bin_sums),
                          where=np.bincount(bin_indices)!=0)
    return bin_means


def load_spikes(session_path, probe, only_good=True):
    spikes = dict()
    spikes['times'] = np.load(join(session_path, probe, 'spikes.times.npy'))
    spikes['distances'] = np.load(join(session_path, probe, 'spikes.distances.npy'))
    spikes['clusters'] = np.load(join(session_path, probe, 'spikes.clusters.npy'))
    clusters = dict()
    clusters['bc_label'] = np.load(join(session_path, probe, 'clusters.bcUnitType.npy'),
                                   allow_pickle=True)
    clusters['ks_label'] = pd.read_csv(join(session_path, probe, 'cluster_KSLabel.tsv'),
                                       sep='\t')['KSLabel']
    if isfile(join(session_path, probe, 'cluster_group.tsv')):
        clusters['manual_label'] = pd.read_csv(join(session_path, probe, 'cluster_group.tsv'),
                                               sep='\t')['group']
    if only_good:
        good_units = np.where(clusters['manual_label'] == 'good')[0]
        spikes['times'] = spikes['times'][np.isin(spikes['clusters'], good_units)]
        spikes['distances'] = spikes['distances'][np.isin(spikes['clusters'], good_units)]
        spikes['clusters'] = spikes['clusters'][np.isin(spikes['clusters'], good_units)]
    return spikes, clusters


def peri_event_trace(array, timestamps, event_times, event_ids, ax, t_before=1, t_after=3,
                     event_labels=None, color_palette='colorblind', ind_lines=False, kwargs=[]):

    # Construct dataframe for plotting
    plot_df = pd.DataFrame()
    samp_rate = np.round(np.mean(np.diff(timestamps)), 3)
    time_x = np.arange(0, t_before + t_after, samp_rate)
    time_x = time_x - time_x[int(t_before * (1/samp_rate))]
    for i, t in enumerate(event_times[~np.isnan(event_times)]):
        zero_point = np.argmin(np.abs(timestamps - t))
        this_array = array[zero_point - np.sum(time_x < 0) : (zero_point + np.sum(time_x > 0)) + 1]
        if this_array.shape[0] != time_x.shape[0]:
            print('Trial time mismatch')
            continue
        plot_df = pd.concat((plot_df, pd.DataFrame(data={
            'y': this_array, 'time': time_x, 'event_id': event_ids[i], 'event_nr': i})))

    # Plot
    if ind_lines:
        sns.lineplot(data=plot_df, x='time', y='y', hue='event_id', estimator=None, units='event_nr',
                     palette=color_palette, err_kws={'lw': 0}, ax=ax, **kwargs)
    else:
        sns.lineplot(data=plot_df, x='time', y='y', hue='event_id', errorbar='se',
                     palette=color_palette, err_kws={'lw': 0}, ax=ax, **kwargs)
    if event_labels is None:
        ax.get_legend().remove()
    else:
        g = ax.legend(title='', prop={'size': 5.5})
        for t, l in zip(g.texts, event_labels):
            t.set_text(l)

    # sns.despine(trim=True)
    # plt.tight_layout()


def calculate_peths(
        spike_times, spike_clusters, cluster_ids, align_times, pre_time=0.2,
        post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True):
    """
    From ibllib package    

    Calcluate peri-event time histograms; return means and standard deviations
    for each time point across specified clusters

    :param spike_times: spike times (in seconds)
    :type spike_times: array-like
    :param spike_clusters: cluster ids corresponding to each event in `spikes`
    :type spike_clusters: array-like
    :param cluster_ids: subset of cluster ids for calculating peths
    :type cluster_ids: array-like
    :param align_times: times (in seconds) to align peths to
    :type align_times: array-like
    :param pre_time: time (in seconds) to precede align times in peth
    :type pre_time: float
    :param post_time: time (in seconds) to follow align times in peth
    :type post_time: float
    :param bin_size: width of time windows (in seconds) to bin spikes
    :type bin_size: float
    :param smoothing: standard deviation (in seconds) of Gaussian kernel for
        smoothing peths; use `smoothing=0` to skip smoothing
    :type smoothing: float
    :param return_fr: `True` to return (estimated) firing rate, `False` to return spike counts
    :type return_fr: bool
    :return: peths, binned_spikes
    :rtype: peths: Bunch({'mean': peth_means, 'std': peth_stds, 'tscale': ts, 'cscale': ids})
    :rtype: binned_spikes: np.array (n_align_times, n_clusters, n_bins)
    """

    # initialize containers
    n_offset = 5 * int(np.ceil(smoothing / bin_size))  # get rid of boundary effects for smoothing
    n_bins_pre = int(np.ceil(pre_time / bin_size)) + n_offset
    n_bins_post = int(np.ceil(post_time / bin_size)) + n_offset
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), n_bins))

    # build gaussian kernel if requested
    if smoothing > 0:
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)
        # half (causal) gaussian filter
        # window[int(np.ceil(w/2)):] = 0
        window /= np.sum(window)
        binned_spikes_conv = np.copy(binned_spikes)

    ids = np.unique(cluster_ids)

    # filter spikes outside of the loop
    idxs = np.bitwise_and(spike_times >= np.min(align_times) - (n_bins_pre + 1) * bin_size,
                          spike_times <= np.max(align_times) + (n_bins_post + 1) * bin_size)
    idxs = np.bitwise_and(idxs, np.isin(spike_clusters, cluster_ids))
    spike_times = spike_times[idxs]
    spike_clusters = spike_clusters[idxs]

    # compute floating tscale
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    # bin spikes
    for i, t_0 in enumerate(align_times):
        # define bin edges
        ts = tscale + t_0
        # filter spikes
        idxs = np.bitwise_and(spike_times >= ts[0], spike_times <= ts[-1])
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]

        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        xscale = ts
        xind = (np.floor((i_spikes - np.min(ts)) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)

        # store (ts represent bin edges, so there are one fewer bins)
        bs_idxs = np.isin(ids, yscale)
        binned_spikes[i, bs_idxs, :] = r[:, :-1]

        # smooth
        if smoothing > 0:
            idxs = np.where(bs_idxs)[0]
            for j in range(r.shape[0]):
                binned_spikes_conv[i, idxs[j], :] = convolve(
                    r[j, :], window, mode='same', method='auto')[:-1]

    # average
    if smoothing > 0:
        binned_spikes_ = np.copy(binned_spikes_conv)
    else:
        binned_spikes_ = np.copy(binned_spikes)
    if return_fr:
        binned_spikes_ /= bin_size

    peth_means = np.mean(binned_spikes_, axis=0)
    peth_stds = np.std(binned_spikes_, axis=0)

    if smoothing > 0:
        peth_means = peth_means[:, n_offset:-n_offset]
        peth_stds = peth_stds[:, n_offset:-n_offset]
        binned_spikes = binned_spikes[:, :, n_offset:-n_offset]
        tscale = tscale[n_offset:-n_offset]

    # package output
    tscale = (tscale[:-1] + tscale[1:]) / 2
    peths = dict({'means': peth_means, 'stds': peth_stds, 'tscale': tscale, 'cscale': ids})
    return peths, binned_spikes


def peri_multiple_events_time_histogram(
        spike_times, spike_clusters, events, event_ids, cluster_id,
        t_before=0.2, t_after=0.5, bin_size=0.025, smoothing=0.025, as_rate=True,
        include_raster=False, error_bars='sem', ax=None,
        pethline_kwargs=[{'color': 'blue', 'lw': 2}, {'color': 'red', 'lw': 2}],
        errbar_kwargs=[{'color': 'blue', 'alpha': 0.5}, {'color': 'red', 'alpha': 0.5}],
        raster_kwargs=[{'color': 'blue', 'lw': 0.5}, {'color': 'red', 'lw': 0.5}],
        eventline_kwargs={'color': 'black', 'alpha': 0.5}, **kwargs):
    """
    From ibllib package

    Plot peri-event time histograms, with the meaning firing rate of units centered on a given
    series of events. Can optionally add a raster underneath the PETH plot of individual spike
    trains about the events.

    Parameters
    ----------
    spike_times : array_like
        Spike times (in seconds)
    spike_clusters : array-like
        Cluster identities for each element of spikes
    events : array-like
        Times to align the histogram(s) to
    event_ids : array-like
        Identities of events
    cluster_id : int
        Identity of the cluster for which to plot a PETH

    t_before : float, optional
        Time before event to plot (default: 0.2s)
    t_after : float, optional
        Time after event to plot (default: 0.5s)
    bin_size :float, optional
        Width of bin for histograms (default: 0.025s)
    smoothing : float, optional
        Sigma of gaussian smoothing to use in histograms. (default: 0.025s)
    as_rate : bool, optional
        Whether to use spike counts or rates in the plot (default: `True`, uses rates)
    include_raster : bool, optional
        Whether to put a raster below the PETH of individual spike trains (default: `False`)
    error_bars : {'std', 'sem', 'none'}, optional
        Defines which type of error bars to plot. Options are:
        -- `'std'` for 1 standard deviation
        -- `'sem'` for standard error of the mean
        -- `'none'` for only plotting the mean value
        (default: `'std'`)
    ax : matplotlib axes, optional
        If passed, the function will plot on the passed axes. Note: current
        behavior causes whatever was on the axes to be cleared before plotting!
        (default: `None`)
    pethline_kwargs : dict, optional
        Dict containing line properties to define PETH plot line. Default
        is a blue line with weight of 2. Needs to have color. See matplotlib plot documentation
        for more options.
        (default: `{'color': 'blue', 'lw': 2}`)
    errbar_kwargs : dict, optional
        Dict containing fill-between properties to define PETH error bars.
        Default is a blue fill with 50 percent opacity.. Needs to have color. See matplotlib
        fill_between documentation for more options.
        (default: `{'color': 'blue', 'alpha': 0.5}`)
    eventline_kwargs : dict, optional
        Dict containing fill-between properties to define line at event.
        Default is a black line with 50 percent opacity.. Needs to have color. See matplotlib
        vlines documentation for more options.
        (default: `{'color': 'black', 'alpha': 0.5}`)
    raster_kwargs : dict, optional
        Dict containing properties defining lines in the raster plot.
        Default is black lines with line width of 0.5. See matplotlib vlines for more options.
        (default: `{'color': 'black', 'lw': 0.5}`)

    Returns
    -------
        ax : matplotlib axes
            Axes with all of the plots requested.
    """

    # Check to make sure if we fail, we fail in an informative way
    if not len(spike_times) == len(spike_clusters):
        raise ValueError('Spike times and clusters are not of the same shape')
    if len(events) == 1:
        raise ValueError('Cannot make a PETH with only one event.')
    if error_bars not in ('std', 'sem', 'none'):
        raise ValueError('Invalid error bar type was passed.')
    if not all(np.isfinite(events)):
        raise ValueError('There are NaN or inf values in the list of events passed. '
                         ' Please remove non-finite data points and try again.')

    # Construct an axis object if none passed
    if ax is None:
        plt.figure()
        ax = plt.gca()
    # Plot the curves and add error bars
    mean_max, bars_max = [], []
    for i, event_id in enumerate(np.unique(event_ids)):
        # Compute peths
        peths, binned_spikes = singlecell.calculate_peths(spike_times, spike_clusters, [cluster_id],
                                                          events[event_ids == event_id], t_before,
                                                          t_after, bin_size, smoothing, as_rate)
        mean = peths.means[0, :]
        ax.plot(peths.tscale, mean, **pethline_kwargs[i])
        if error_bars == 'std':
            bars = peths.stds[0, :]
        elif error_bars == 'sem':
            bars = peths.stds[0, :] / np.sqrt(np.sum(event_ids == event_id))
        else:
            bars = np.zeros_like(mean)
        if error_bars != 'none':
            ax.fill_between(peths.tscale, mean - bars, mean + bars, **errbar_kwargs[i])
        mean_max.append(mean.max())
        bars_max.append(bars[mean.argmax()])

    # Plot the event marker line. Extends to 5% higher than max value of means plus any error bar.
    plot_edge = (np.max(mean_max) + bars_max[np.argmax(mean_max)]) * 1.05
    ax.vlines(0., 0., plot_edge, **eventline_kwargs)
    # Set the limits on the axes to t_before and t_after. Either set the ylim to the 0 and max
    # values of the PETH, or if we want to plot a spike raster below, create an equal amount of
    # blank space below the zero where the raster will go.
    ax.set_xlim([-t_before, t_after])
    ax.set_ylim([-plot_edge if include_raster else 0., plot_edge])
    # Put y ticks only at min, max, and zero
    if mean.min() != 0:
        ax.set_yticks([0, mean.min(), mean.max()])
    else:
        ax.set_yticks([0., mean.max()])
    # Move the x axis line from the bottom of the plotting space to zero if including a raster,
    # Then plot the raster
    if include_raster:
        ax.axhline(0., color='black', lw=0.5)
        tickheight = plot_edge / len(events)  # How much space per trace
        tickedges = np.arange(0., -plot_edge - 1e-5, -tickheight)
        clu_spks = spike_times[spike_clusters == cluster_id]
        ii = 0
        for k, event_id in enumerate(np.unique(event_ids)):
            for i, t in enumerate(events[event_ids == event_id]):
                idx = np.bitwise_and(clu_spks >= t - t_before, clu_spks <= t + t_after)
                event_spks = clu_spks[idx]
                ax.vlines(event_spks - t, tickedges[i + ii + 1], tickedges[i + ii],
                          **raster_kwargs[k])
            ii += np.sum(event_ids == event_id)
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes', y=0.75)
    else:
        ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s) after event')
    return ax
