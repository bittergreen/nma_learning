import numpy as np
import matplotlib.pyplot as plt
# @title Data retrieval and loading
import os
import hashlib
import requests

from scipy.optimize import minimize
from scipy.io import loadmat


def download_data():
    fname = "RGCdata.mat"
    url = "https://osf.io/mzujs/download"
    expected_md5 = "1b2977453020bce5319f2608c94d38d0"

    if not os.path.isfile(fname):
        try:
            r = requests.get(url)
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("!!! Failed to download data !!!")
            elif hashlib.md5(r.content).hexdigest() != expected_md5:
                print("!!! Data download appears corrupted !!!")
            else:
                with open(fname, "wb") as fid:
                    fid.write(r.content)


# @title Plotting Functions

def plot_stim_and_spikes(stim, spikes, dt, nt=120):
    """Show time series of stim intensity and spike counts.

  Args:
    stim (1D array): vector of stimulus intensities
    spikes (1D array): vector of spike counts
    dt (number): duration of each time step
    nt (number): number of time steps to plot

  """
    timepoints = np.arange(nt)
    time = timepoints * dt

    f, (ax_stim, ax_spikes) = plt.subplots(
        nrows=2, sharex=True, figsize=(8, 5),
    )
    ax_stim.plot(time, stim[timepoints])
    ax_stim.set_ylabel('Stimulus intensity')

    ax_spikes.plot(time, spikes[timepoints])
    ax_spikes.set_xlabel('Time (s)')
    ax_spikes.set_ylabel('Number of spikes')

    f.tight_layout()
    plt.show()


def plot_glm_matrices(X, y, nt=50):
    """Show X and Y as heatmaps.

  Args:
    X (2D array): Design matrix.
    y (1D or 2D array): Target vector.

  """
    from matplotlib.colors import BoundaryNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    Y = np.c_[y]  # Ensure Y is 2D and skinny

    f, (ax_x, ax_y) = plt.subplots(
        ncols=2,
        figsize=(6, 8),
        sharey=True,
        gridspec_kw=dict(width_ratios=(5, 1)),
    )
    norm = BoundaryNorm([-1, -.2, .2, 1], 256)
    imx = ax_x.pcolormesh(X[:nt], cmap="coolwarm", norm=norm)

    ax_x.set(
        title="X\n(lagged stimulus)",
        xlabel="Time lag (time bins)",
        xticks=[4, 14, 24],
        xticklabels=['-20', '-10', '0'],
        ylabel="Time point (time bins)",
    )
    plt.setp(ax_x.spines.values(), visible=True)

    divx = make_axes_locatable(ax_x)
    caxx = divx.append_axes("right", size="5%", pad=0.1)
    cbarx = f.colorbar(imx, cax=caxx)
    cbarx.set_ticks([-.6, 0, .6])
    cbarx.set_ticklabels(np.sort(np.unique(X)))

    norm = BoundaryNorm(np.arange(y.max() + 1), 256)
    imy = ax_y.pcolormesh(Y[:nt], cmap="magma", norm=norm)
    ax_y.set(
        title="Y\n(spike count)",
        xticks=[]
    )
    ax_y.invert_yaxis()
    plt.setp(ax_y.spines.values(), visible=True)

    divy = make_axes_locatable(ax_y)
    caxy = divy.append_axes("right", size="30%", pad=0.1)
    cbary = f.colorbar(imy, cax=caxy)
    cbary.set_ticks(np.arange(y.max()) + .5)
    cbary.set_ticklabels(np.arange(y.max()))
    plt.show()


def plot_spike_filter(theta, dt, show=True, **kws):
    """Plot estimated weights based on time lag model.

  Args:
    theta (1D array): Filter weights, not including DC term.
    dt (number): Duration of each time bin.
    kws: Pass additional keyword arguments to plot()
    show (boolean): To plt.show or not the plot.
  """
    d = len(theta)
    t = np.arange(-d + 1, 1) * dt

    ax = plt.gca()
    ax.plot(t, theta, marker="o", **kws)
    ax.axhline(0, color=".2", linestyle="--", zorder=1)
    ax.set(
        xlabel="Time before spike (s)",
        ylabel="Filter weight",
    )
    if show:
        plt.show()


def plot_spikes_with_prediction(spikes, predicted_spikes, dt,
                                nt=50, t0=120, **kws):
    """Plot actual and predicted spike counts.

  Args:
    spikes (1D array): Vector of actual spike counts
    predicted_spikes (1D array): Vector of predicted spike counts
    dt (number): Duration of each time bin.
    nt (number): Number of time bins to plot
    t0 (number): Index of first time bin to plot.
    show (boolean): To plt.show or not the plot.
    kws: Pass additional keyword arguments to plot()

  """
    t = np.arange(t0, t0 + nt) * dt

    f, ax = plt.subplots()
    lines = ax.stem(t, spikes[:nt])
    plt.setp(lines, color=".5")
    lines[-1].set_zorder(1)
    kws.setdefault("linewidth", 3)
    yhat, = ax.plot(t, predicted_spikes[:nt], **kws)
    ax.set(
        xlabel="Time (s)",
        ylabel="Spikes",
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend([lines[0], yhat], ["Spikes", "Predicted"])
    plt.show()



# @title Plotting Functions

def plot_weights(models, sharey=True):
    """Draw a stem plot of weights for each model in models dict."""
    n = len(models)
    f = plt.figure(figsize=(10, 2.5 * n))
    axs = f.subplots(n, sharex=True, sharey=sharey)
    axs = np.atleast_1d(axs)

    for ax, (title, model) in zip(axs, models.items()):
        ax.margins(x=.02)
        stem = ax.stem(model.coef_.squeeze())
        stem[0].set_marker(".")
        stem[0].set_color(".2")
        stem[1].set_linewidths(.5)
        stem[1].set_color(".2")
        stem[2].set_visible(False)
        ax.axhline(0, color="C3", lw=3)
        ax.set(ylabel="Weight", title=title)
    ax.set(xlabel="Neuron (a.k.a. feature)")
    f.tight_layout()
    plt.show()


def plot_function(f, name, var, points=(-10, 10)):
    """Evaluate f() on linear space between points and plot.

    Args:
      f (callable): function that maps scalar -> scalar
      name (string): Function name for axis labels
      var (string): Variable name for axis labels.
      points (tuple): Args for np.linspace to create eval grid.
    """
    x = np.linspace(*points)
    ax = plt.figure().subplots()
    ax.plot(x, f(x))
    ax.set(
        xlabel=f'${var}$',
        ylabel=f'${name}({var})$'
    )
    plt.show()


def plot_model_selection(C_values, accuracies):
    """Plot the accuracy curve over log-spaced C values."""
    ax = plt.figure().subplots()
    ax.set_xscale("log")
    ax.plot(C_values, accuracies, marker="o")
    best_C = C_values[np.argmax(accuracies)]
    ax.set(
        xticks=C_values,
        xlabel="C",
        ylabel="Cross-validated accuracy",
        title=f"Best C: {best_C:1g} ({np.max(accuracies):.2%})",
    )
    plt.show()


def plot_non_zero_coefs(C_values, non_zero_l1, n_voxels):
    """Plot the accuracy curve over log-spaced C values."""
    ax = plt.figure().subplots()
    ax.set_xscale("log")
    ax.plot(C_values, non_zero_l1, marker="o")
    ax.set(
        xticks=C_values,
        xlabel="C",
        ylabel="Number of non-zero coefficients",
    )
    ax.axhline(n_voxels, color=".1", linestyle=":")
    ax.annotate("Total\n# Neurons", (C_values[0], n_voxels * .98), va="top")
    plt.show()


def download_steinmetz_data():
    url = "https://osf.io/r9gh8/download"
    fname = "W1D4_steinmetz_data.npz"
    expected_md5 = "d19716354fed0981267456b80db07ea8"

    if not os.path.isfile(fname):
        try:
            r = requests.get(url)
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("!!! Failed to download data !!!")
            elif hashlib.md5(r.content).hexdigest() != expected_md5:
                print("!!! Data download appears corrupted !!!")
            else:
                with open(fname, "wb") as fid:
                    fid.write(r.content)


def load_steinmetz_data():
    download_steinmetz_data()
    fname = "W1D4_steinmetz_data.npz"
    with np.load(fname) as dobj:
        data = dict(**dobj)
    return data

