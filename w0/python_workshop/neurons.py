import numpy as np
import matplotlib.pyplot as plt


def plot_all(t_range, v, raster=None, spikes=None, spikes_mean=None):
    # Initialize the figure
    plt.figure()

    # Plot simulations and sample mean
    ax1 = plt.subplot(4, 1, 1)
    for j in range(n):
        plt.scatter(t_range, v[j], color="k", marker=".", alpha=0.01)
    plt.plot(t_range, np.mean(v, axis=0), 'C1', alpha=0.8, linewidth=3)
    plt.ylabel('$V_m$ (V)')

    # Plot spikes
    plt.subplot(4, 1, 2, sharex=ax1)
    # for each neuron j: collect spike times and plot them at height j
    for j in range(n):
        times = np.array(spikes[j])
        plt.scatter(times, j * np.ones_like(times), color="C0", marker='.', alpha=0.2)

    plt.ylabel('neuron')

    # Plot firing rate
    plt.subplot(4, 1, 3, sharex=ax1)
    plt.plot(t_range, spikes_mean)
    plt.xlabel('time (s)')
    plt.ylabel('rate (Hz)')

    plt.tight_layout()

    plt.show()


# Simulation class
class LIFNeurons:
    """
  Keep track of membrane potential for multiple realizations of LIF neuron,
  and performs single step discrete time integration.
  """

    def __init__(self, n, t_ref_mu=0.01, t_ref_sigma=0.002,
                 tau=20e-3, el=-60e-3, vr=-70e-3, vth=-50e-3, r=100e6):
        # Neuron count
        self.n = n

        # Neuron parameters
        self.tau = tau  # second
        self.el = el  # milivolt
        self.vr = vr  # milivolt
        self.vth = vth  # milivolt
        self.r = r  # ohm

        # Initializes refractory period distribution
        self.t_ref_mu = t_ref_mu
        self.t_ref_sigma = t_ref_sigma
        self.t_ref = self.t_ref_mu + self.t_ref_sigma * np.random.normal(size=self.n)
        self.t_ref[self.t_ref < 0] = 0

        # State variables
        self.v = self.el * np.ones(self.n)
        self.spiked = self.v >= self.vth
        self.last_spike = -self.t_ref * np.ones([self.n])
        self.t = 0.
        self.steps = 0

    def ode_step(self, dt, i):
        # Update running time and steps
        self.t += dt
        self.steps += 1

        # One step of discrete time integration of dt
        self.v = self.v + dt / self.tau * (self.el - self.v + self.r * i)

        # Spike and clamp
        self.spiked = (self.v[step] >= vth)
        self.v[self.spiked] = vr

        clamped = (self.last_spike + self.t_ref > self.t)
        self.v[clamped] = vr

        self.last_spike[self.spiked] = self.t


# Set random number generator
np.random.seed(2020)

t_max = 150e-3  # second
dt = 1e-3  # second
tau = 20e-3  # second
el = -60e-3  # milivolt
vr = -70e-3  # milivolt
vth = -50e-3  # milivolt
r = 100e6  # ohm
i_mean = 25e-11  # ampere

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt) ** 0.5 * (2 * np.random.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n, step_end])

# Initialize neurons
neurons = LIFNeurons(n)

# Loop over time steps
for step, t in enumerate(t_range):
    # Call ode_step method
    neurons.ode_step(dt, i[:, step])

    # Log v_n and spike history
    v_n[:, step] = neurons.v
    raster[neurons.spiked, step] = 1.

# Report running time and steps
print(f'Ran for {neurons.t:.3}s in {neurons.steps} steps.')

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, raster)
