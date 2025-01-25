import numpy as np
import matplotlib.pyplot as plt

t_max = 150e-3  # second
dt = 1e-3  # second
tau = 20e-3  # second
el = -60e-3  # milivolt
vr = -70e-3  # milivolt
vth = -50e-3  # milivolt
r = 100e6  # ohm
i_mean = 25e-11  # ampere


def plot_all(t_range, v, raster=None, spikes=None, spikes_mean=None):
    # Initialize the figure
    plt.figure()

    # Plot simulations and sample mean
    ax1 = plt.subplot(3, 1, 1)
    for j in range(n):
        plt.scatter(t_range, v[j], color="k", marker=".", alpha=0.01)
    plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
    plt.ylabel('$V_m$ (V)')

    # Plot spikes
    plt.subplot(3, 1, 2, sharex=ax1)
    # for each neuron j: collect spike times and plot them at height j
    for j in range(n):
        times = np.array(spikes[j])
        plt.scatter(times, j * np.ones_like(times), color="C0", marker='.', alpha=0.2)

    plt.ylabel('neuron')

    # Plot firing rate
    plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(t_range, spikes_mean)
    plt.xlabel('time (s)')
    plt.ylabel('rate (Hz)')

    plt.tight_layout()

    plt.show()


# Set random number generator
np.random.seed(2020)

# Initialize t_range, step_end, n, v_n, i and nbins
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500  # num of trials
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt) ** 0.5 * (2 * np.random.random([n, step_end]) - 1))
# nbins = 50

# Initialize spikes and spikes_n
spikes = {j: [] for j in range(n)}
spikes_n = np.zeros([step_end])

# Initialize binary numpy array for raster plot
raster = np.zeros([n, step_end])

# Initialize t_ref and last_spike
mu = 0.01
sigma = 0.007
t_ref = mu + sigma*np.random.normal(size=n)
t_ref[t_ref<0] = 0
last_spike = -t_ref * np.ones([n])


# Loop over time steps
for step, t in enumerate(t_range):

    # Skip first iteration
    if step == 0:
        continue

    # Compute v_n
    v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

    # Initialize boolean numpy array clamped using last_spike, t and t_ref
    clamped = (last_spike + t_ref > t)

    # Reset clamped neurons to vr using clamped
    v_n[clamped, step] = vr

    # Initialize boolean numpy array `spiked` with v_n > v_thr
    spiked = (v_n[:, step] >= vth)

    # Set relevant values of v_n to resting potential using spiked
    v_n[spiked, step] = vr

    # Set relevant elements in raster to 1 using spiked
    raster[spiked, step] = 1.

    # Update numpy array last_spike with time t for spiking neurons
    last_spike[spiked] = t

    # Collect spike times
    for j in np.where(spiked)[0]:
        spikes[j] += [t]
        spikes_n[step] += 1


# Collect mean Vm and mean spiking rate
v_mean = np.mean(v_n, axis=0)
spikes_mean = spikes_n / n


plot_all(t_range, v_n, spikes=spikes, spikes_mean=spikes_mean)

# plot histogram of t_ref
plt.figure(figsize=(8,4))
plt.hist(t_ref, bins=32, histtype='stepfilled', linewidth=0, color='C1')
plt.xlabel(r'$t_{ref}$ (s)')
plt.ylabel('count')
plt.tight_layout()

plt.show()

"""

# Initialize the figure
plt.figure()
plt.ylabel('Frequency')
plt.xlabel('$V_m$ (V)')

# Plot a histogram at t_max/10 (add labels and parameters histtype='stepfilled' and linewidth=0)
plt.hist(v_n[:, int(step_end/10)], nbins, histtype='stepfilled', linewidth=0, label=f't={t_max/10} s')

# Plot a histogram at t_max (add labels and parameters histtype='stepfilled' and linewidth=0)
plt.hist(v_n[:, -1], nbins, histtype='stepfilled', linewidth=0, label=f't={t_max} s')

# Add legend
plt.legend()
plt.show()


# initialize the figure
plt.figure()
"""
