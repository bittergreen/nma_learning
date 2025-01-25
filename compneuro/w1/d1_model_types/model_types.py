import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils import *


def plot_data(spike_times):
    # plot the distribution of total spike times of each neuron in the data
    total_spikes_per_neuron = [len(spike_times[i]) for i in range(len(spike_times))]
    mean_spike_count = np.mean(total_spikes_per_neuron)
    median_spike_count = np.median(total_spikes_per_neuron)  # Hint: Try the function np.median
    plt.hist(total_spikes_per_neuron, bins=50)
    plt.xlabel("Total spikes per neuron")
    plt.ylabel("Number of neurons")
    plt.axvline(median_spike_count, color="limegreen", label="Median neuron")
    plt.axvline(mean_spike_count, color="orange", label="Mean neuron")
    plt.legend()
    plt.show()
    frac_below_mean = (total_spikes_per_neuron < mean_spike_count).mean()
    print(f"{frac_below_mean:2.1%} of neurons are below the mean")


def restrict_spike_times(spike_times, interval):
    """Given a spike_time dataset, restrict to spikes within given interval.

    Args:
        spike_times (sequence of np.ndarray): List or array of arrays,
          each inner array has spike times for a single neuron.
        interval (tuple): Min, max time values; keep min <= t < max.

    Returns:
        np.ndarray: like `spike_times`, but only within `interval`
    """
    interval_spike_times = []
    for spikes in spike_times:
        interval_mask = (spikes >= interval[0]) & (spikes < interval[1])
        interval_spike_times.append(spikes[interval_mask])
    return np.array(interval_spike_times, object)


def get_max_interval(data_arr):
    # return the (max - min) of recorded spiking time in the whole data array
    data_flat = np.concatenate(data_arr, 0)
    max_interval = np.ptp(data_flat)
    return max_interval


def plot_spikes(data_arr, neuron_idx):
    # plot the spikes of indexed neurons in the data array
    plt.eventplot(data_arr[neuron_idx], color=".2")
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.show()


def exponential(xs, scale, rate, x0):
    """A simple parameterized exponential function, applied element-wise.

  Args:
    xs (np.ndarray or float): Input(s) to the function.
    scale (float): Linear scaling factor.
    rate (float): Exponential growth (positive) or decay (negative) rate.
    x0 (float): Horizontal offset.

  """
    ys = scale * np.exp(rate * (xs - x0))
    return ys


def inverse(xs, scale, x0):
    """A simple parameterized inverse function (`1/x`), applied element-wise.

  Args:
    xs (np.ndarray or float): Input(s) to the function.
    scale (float): Linear scaling factor.
    x0 (float): Horizontal offset.

  """
    ys = scale / (xs - x0)
    return ys


def linear(xs, slope, y0):
    """A simple linear function, applied element-wise.

  Args:
    xs (np.ndarray or float): Input(s) to the function.
    slope (float): Slope of the line.
    y0 (float): y-intercept of the line.

  """
    ys = slope * xs + y0
    return ys


def compute_single_neuron_isis(data_arr, neuron_idx):
    """Compute a vector of ISIs for a single neuron given spike times.

      Args:
        data_arr (list of 1D arrays): Spike time dataset, with the first
          dimension corresponding to different neurons.
        neuron_idx (int): Index of the unit to compute ISIs for.

      Returns:
        isis (1D array): Duration of time between each spike from one neuron.
    """

    # Extract the spike times for the specified neuron
    single_neuron_spikes = data_arr[neuron_idx]

    # Compute the ISIs for this set of spikes
    # Hint: the function np.diff computes discrete differences along an array
    isis = np.diff(single_neuron_spikes, axis=0)

    return isis


def fit_plot(
        exp_scale=5000, exp_rate=-10, exp_x0=0.01,
        inv_scale=1000, inv_x0=0,
        lin_slope=-1e5, lin_y0=2000,
):
    """Helper function for plotting function fits with interactive sliders."""
    func_params = dict(
        exponential=(exp_scale, exp_rate, exp_x0),
        inverse=(inv_scale, inv_x0),
        linear=(lin_slope, lin_y0),
    )
    single_neuron_idx = 283
    single_neuron_spikes = spike_times[single_neuron_idx]
    single_neuron_isis = np.diff(single_neuron_spikes)

    counts, edges = np.histogram(
        single_neuron_isis,
        bins=50,
        range=(0, single_neuron_isis.max())
    )

    functions = dict(
        exponential=exponential,
        inverse=inverse,
        linear=linear,
    )

    colors = dict(
        exponential="C1",
        inverse="C2",
        linear="C4",
    )

    f, ax = plt.subplots()
    ax.fill_between(edges[:-1], counts, step="post", alpha=.5)
    xs = np.linspace(1e-10, edges.max())
    for name, function in functions.items():
        ys = function(xs, *func_params[name])
        ax.plot(xs, ys, lw=3, color=colors[name], label=name);
    ax.set(
        xlim=(edges.min(), edges.max()),
        ylim=(0, counts.max() * 1.1),
        xlabel="ISI (s)",
        ylabel="Number of spikes",
    )
    ax.legend()
    plt.show()


def lif_neuron(n_steps=1000, alpha=0.01, rate=10):
    """ Simulate a linear integrate-and-fire neuron.

  Args:
    n_steps (int): The number of time steps to simulate the neuron's activity.
    alpha (float): The input scaling factor
    rate (int): The mean rate of incoming spikes

  """
    # Precompute Poisson samples for speed
    exc = stats.poisson(rate).rvs(n_steps)

    # Initialize voltage and spike storage
    v = np.zeros(n_steps)
    spike_times = []

    # Loop over steps
    for i in range(1, n_steps):

        # Update v
        dv = alpha * exc[i]
        v[i] = v[i - 1] + dv

        # If spike happens, reset voltage and record
        if v[i] > 1:
            spike_times.append(i)
            v[i] = 0

    return v, spike_times


def lif_neuron_inh(n_steps=1000, alpha=0.5, beta=0.1, exc_rate=10, inh_rate=10):
    """ Simulate a simplified leaky integrate-and-fire neuron with both excitatory
  and inhibitory inputs.

  Args:
    n_steps (int): The number of time steps to simulate the neuron's activity.
    alpha (float): The input scaling factor
    beta (float): The membrane potential leakage factor
    exc_rate (int): The mean rate of the incoming excitatory spikes
    inh_rate (int): The mean rate of the incoming inhibitory spikes
  """

    # precompute Poisson samples for speed
    exc = stats.poisson(exc_rate).rvs(n_steps)
    inh = stats.poisson(inh_rate).rvs(n_steps)

    v = np.zeros(n_steps)
    spike_times = []

    for i in range(1, n_steps):

        dv = -beta * v[i - 1] + alpha * (exc[i] - inh[i])

        v[i] = v[i - 1] + dv
        if v[i] > 1:
            spike_times.append(i)
            v[i] = 0

    return v, spike_times


def entropy(pmf):
    """Given a discrete distribution, return the Shannon entropy in bits.

      This is a measure of information in the distribution. For a totally
      deterministic distribution, where samples are always found in the same bin,
      then samples from the distribution give no more information and the entropy
      is 0.

      For now this assumes `pmf` arrives as a well-formed distribution (that is,
      `np.sum(pmf)==1` and `not np.any(pmf < 0)`)

      Args:
        pmf (np.ndarray): The probability mass function for a discrete distribution
          represented as an array of probabilities.
      Returns:
        h (number): The entropy of the distribution in `pmf`.

      """
    # reduce to non-zero entries to avoid an error from log2(0)
    pmf = pmf[pmf > 0]

    # implement the equation for Shannon entropy (in bits)
    h = -np.sum(pmf * np.log2(pmf))

    # return the absolute value (avoids getting a -0 result)
    return np.abs(h)


def unit_pmf():
    n_bins = 50  # number of points supporting the distribution
    x_range = (0, 1)  # will be subdivided evenly into bins corresponding to points

    pmf = np.zeros(n_bins)
    pmf[len(pmf) // 2] = 1.0  # middle point has all the mass

    # Since we already have a PMF, rather than un-binned samples, `plt.hist` is not
    # suitable. Instead, we directly plot the PMF as a step function to visualize
    # the histogram:
    pmf_ = np.insert(pmf, 0, pmf[0])  # this is necessary to align plot steps with bin edges

    return pmf_


def compare_distributions(n_bins, mean_isi, isi_range):

    bins = np.linspace(*isi_range, n_bins + 1)
    mean_idx = np.searchsorted(bins, mean_isi)

    # 1. all mass concentrated on the ISI mean
    pmf_single = np.zeros(n_bins)
    # pmf_single[mean_idx] = 1.0

    # 2. mass uniformly distributed about the ISI mean
    pmf_uniform = np.zeros(n_bins)
    pmf_uniform[0:2 * mean_idx] = 1 / (2 * mean_idx)

    # 3. mass exponentially distributed about the ISI mean
    pmf_exp = stats.expon.pdf(bins[1:], scale=mean_isi)
    pmf_exp /= np.sum(pmf_exp)

    print(
        f"Deterministic: {entropy(pmf_single):.2f} bits",
        f"Uniform: {entropy(pmf_uniform):.2f} bits",
        f"Exponential: {entropy(pmf_exp):.2f} bits",
        sep="\n",
    )


def pmf_from_counts(counts):
    pmf = counts / np.sum(counts)
    return pmf


if __name__ == '__main__':
    spike_times = load_data()
    t_interval = (5, 15)  # units are seconds after start of recording
    # interval_spike_times = restrict_spike_times(spike_times, t_interval)
    # plot_data(interval_spike_times)

    # Compute ISIs
    single_neuron_isis = compute_single_neuron_isis(spike_times, neuron_idx=424)

    # Visualize ISIs
    # plot_isis(single_neuron_isis)
    # fit_plot()

    # Set random seed (for reproducibility)
    np.random.seed(12)

    # Model LIF neuron
    # v, spike_times = lif_neuron(alpha=0.01, rate=10)
    # v, spike_times = lif_neuron_inh(alpha=0.5, beta=0.3, exc_rate=11, inh_rate=10)

    # Visualize
    # plot_neuron_stats(v, spike_times)

    # compare_distributions()

    # Get neuron index
    neuron_idx = 450

    # Get counts of ISIs from Steinmetz data
    isi = np.diff(spike_times[neuron_idx])
    isi_range = (0, 0.25)
    n_bins = 50
    bins = np.linspace(*isi_range, n_bins + 1)
    counts, _ = np.histogram(isi, bins)

    # Compute pmf
    pmf = pmf_from_counts(counts)

    print(f"Entropy for Neuron {neuron_idx}: {entropy(pmf):.2f} bits")

    compare_distributions(n_bins, np.mean(isi), isi_range)

    # Visualize
    # plot_pmf(pmf, isi_range, bins, neuron_idx)


