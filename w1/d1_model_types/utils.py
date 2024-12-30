import os
import io
import requests
import numpy as np
from matplotlib import pyplot as plt


def load_data():
    # check if already downloaded
    file_path = "spike_times.npy"
    if os.path.exists(file_path):
        spike_times = np.load(file_path, allow_pickle=True)
    else:
        r = requests.get('https://osf.io/sy5xt/download')
        if r.status_code != 200:
            raise IOError('Failed to download data')
        else:
            spike_times = np.load(io.BytesIO(r.content), allow_pickle=True)['spike_times']
            np.save(file_path, spike_times)  # save after downloaded

    return spike_times


def plot_isis(single_neuron_isis):
    plt.hist(single_neuron_isis, bins=50, histtype="stepfilled")
    plt.axvline(single_neuron_isis.mean(), color="orange", label="Mean ISI")
    plt.xlabel("ISI duration (s)")
    plt.ylabel("Number of spikes")
    plt.legend()
    plt.show()


# @title Plotting Functions

def histogram(counts, bins, vlines=(), ax=None, ax_args=None, **kwargs):
    """Plot a step histogram given counts over bins."""
    if ax is None:
        _, ax = plt.subplots()

    # duplicate the first element of `counts` to match bin edges
    counts = np.insert(counts, 0, counts[0])

    ax.fill_between(bins, counts, step="pre", alpha=0.4, **kwargs)  # area shading
    ax.plot(bins, counts, drawstyle="steps", **kwargs)  # lines

    for x in vlines:
        ax.axvline(x, color='r', linestyle='dotted')  # vertical line

    if ax_args is None:
        ax_args = {}

    # heuristically set max y to leave a bit of room
    ymin, ymax = ax_args.get('ylim', [None, None])
    if ymax is None:
        ymax = np.max(counts)
        if ax_args.get('yscale', 'linear') == 'log':
            ymax *= 1.5
        else:
            ymax *= 1.1
            if ymin is None:
                ymin = 0

    if ymax == ymin:
        ymax = None

    ax_args['ylim'] = [ymin, ymax]

    ax.set(**ax_args)
    ax.autoscale(enable=False, axis='x', tight=True)


def plot_neuron_stats(v, spike_times):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    # membrane voltage trace
    ax1.plot(v[0:100])
    ax1.set(xlabel='Time', ylabel='Voltage')
    # plot spike events
    for x in spike_times:
        if x >= 100:
            break
        ax1.axvline(x, color='red')

    # ISI distribution
    if len(spike_times) > 1:
        isi = np.diff(spike_times)
        n_bins = np.arange(isi.min(), isi.max() + 2) - .5
        counts, bins = np.histogram(isi, n_bins)
        vlines = []
        if len(isi) > 0:
            vlines = [np.mean(isi)]
        xmax = max(20, int(bins[-1]) + 5)
        histogram(counts, bins, vlines=vlines, ax=ax2, ax_args={
            'xlabel': 'Inter-spike interval',
            'ylabel': 'Number of intervals',
            'xlim': [0, xmax]
        })
    else:
        ax2.set(xlabel='Inter-spike interval',
                ylabel='Number of intervals')
    plt.show()


# @title Plotting Functions

def plot_pmf(pmf, isi_range, bins, neuron_idx):
    """Plot the probability mass function."""
    ymax = max(0.2, 1.05 * np.max(pmf))
    pmf_ = np.insert(pmf, 0, pmf[0])
    plt.plot(bins, pmf_, drawstyle="steps")
    plt.fill_between(bins, pmf_, step="pre", alpha=0.4)
    plt.title(f"Neuron {neuron_idx}")
    plt.xlabel("Inter-spike interval (s)")
    plt.ylabel("Probability mass")
    plt.xlim(isi_range)
    plt.ylim([0, ymax])
    plt.show()
