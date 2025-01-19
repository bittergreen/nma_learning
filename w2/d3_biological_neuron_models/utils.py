# Imports
import numpy as np
import matplotlib.pyplot as plt


# @title Plotting Functions

def plot_volt_trace(pars, v, sp):
    """
  Plot trajetory of membrane potential for a single neuron

  Expects:
  pars   : parameter dictionary
  v      : volt trajetory
  sp     : spike train

  Returns:
  figure of the membrane potential trajetory for a single neuron
  """

    V_th = pars['V_th']
    dt, range_t = pars['dt'], pars['range_t']
    if sp.size:
        sp_num = (sp / dt).astype(int) - 1
        v[sp_num] += 20  # draw nicer spikes

    plt.plot(pars['range_t'], v, 'b')
    plt.axhline(V_th, 0, 1, color='k', ls='--')
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    plt.legend(['Membrane\npotential', r'Threshold V$_{\mathrm{th}}$'],
               loc=[1.05, 0.75])
    plt.ylim([-80, -40])
    plt.show()


def plot_GWN(pars, I_GWN, v, sp):
    """
  Args:
    pars  : parameter dictionary
    I_GWN : Gaussian white noise input

  Returns:
    figure of the gaussian white noise input
  """

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(pars['range_t'][::3], I_GWN[::3], 'b')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$I_{GWN}$ (pA)')
    plt.subplot(122)
    plot_volt_trace(pars, v, sp)
    plt.tight_layout()
    plt.show()


def my_hists(isi1, isi2, cv1, cv2, sigma1, sigma2):
    """
  Args:
    isi1 : vector with inter-spike intervals
    isi2 : vector with inter-spike intervals
    cv1  : coefficient of variation for isi1
    cv2  : coefficient of variation for isi2

  Returns:
    figure with two histograms, isi1, isi2

  """
    plt.figure(figsize=(11, 4))
    my_bins = np.linspace(10, 30, 20)
    plt.subplot(121)
    plt.hist(isi1, bins=my_bins, color='b', alpha=0.5)
    plt.xlabel('ISI (ms)')
    plt.ylabel('count')
    plt.title(r'$\sigma_{GWN}=$%.1f, CV$_{\mathrm{isi}}$=%.3f' % (sigma1, cv1))

    plt.subplot(122)
    plt.hist(isi2, bins=my_bins, color='b', alpha=0.5)
    plt.xlabel('ISI (ms)')
    plt.ylabel('count')
    plt.title(r'$\sigma_{GWN}=$%.1f, CV$_{\mathrm{isi}}$=%.3f' % (sigma2, cv2))
    plt.tight_layout()
    plt.show()


# @title Helper Functions
def default_pars(**kwargs):
    pars = {}

    ### typical neuron parameters###
    pars['V_th'] = -55.  # spike threshold [mV]
    pars['V_reset'] = -75.  # reset potential [mV]
    pars['tau_m'] = 10.  # membrane time constant [ms]
    pars['g_L'] = 10.  # leak conductance [nS]
    pars['V_init'] = -75.  # initial potential [mV]
    pars['V_L'] = -75.  # leak reversal potential [mV]
    pars['tref'] = 2.  # refractory time (ms)

    ### simulation parameters ###
    pars['T'] = 400.  # Total duration of simulation [ms]
    pars['dt'] = .1  # Simulation time step [ms]

    ### external parameters if any ###
    for k in kwargs:
        pars[k] = kwargs[k]

    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # Vector of discretized
    # time points [ms]
    return pars


def run_LIF(pars, Iinj):
    """
  Simulate the LIF dynamics with external input current

  Args:
    pars       : parameter dictionary
    Iinj       : input current [pA]. The injected current here can be a value or an array

  Returns:
    rec_spikes : spike times
    rec_v      : mebrane potential
  """

    # Set parameters
    V_th, V_reset = pars['V_th'], pars['V_reset']
    tau_m, g_L = pars['tau_m'], pars['g_L']
    V_init, V_L = pars['V_init'], pars['V_L']
    dt, range_t = pars['dt'], pars['range_t']
    Lt = range_t.size
    tref = pars['tref']

    # Initialize voltage and current
    v = np.zeros(Lt)
    v[0] = V_init
    Iinj = Iinj * np.ones(Lt)
    tr = 0.

    # simulate the LIF dynamics
    rec_spikes = []  # record spike times
    for it in range(Lt - 1):
        if tr > 0:
            v[it] = V_reset
            tr = tr - 1
        elif v[it] >= V_th:  # reset voltage and record spike event
            rec_spikes.append(it)
            v[it] = V_reset
            tr = tref / dt

        # calculate the increment of the membrane potential
        dv = (-(v[it] - V_L) + Iinj[it] / g_L) * (dt / tau_m)

        # update the membrane potential
        v[it + 1] = v[it] + dv

    rec_spikes = np.array(rec_spikes) * dt

    return v, rec_spikes


def my_GWN(pars, sig, myseed=False):
    """
  Function that calculates Gaussian white noise inputs

  Args:
    pars       : parameter dictionary
    mu         : noise baseline (mean)
    sig        : noise amplitute (standard deviation)
    myseed     : random seed. int or boolean
                 the same seed will give the same random number sequence

  Returns:
    I          : Gaussian white noise input
  """

    # Retrieve simulation parameters
    dt, range_t = pars['dt'], pars['range_t']
    Lt = range_t.size

    # Set random seed. You can fix the seed of the random number generator so
    # that the results are reliable however, when you want to generate multiple
    # realization make sure that you change the seed for each new realization
    if myseed:
        np.random.seed(seed=myseed)
    else:
        np.random.seed()

    # generate GWN
    # we divide here by 1000 to convert units to sec.
    I_GWN = sig * np.random.randn(Lt) * np.sqrt(pars['tau_m'] / dt)

    return I_GWN


def LIF_output_cc(pars, mu, sig, c, bin_size, n_trials=20,
                  correlate_input=None, my_CC=None):
    """ Simulates two LIF neurons with correlated input and computes output correlation

  Args:
  pars       : parameter dictionary
  mu         : noise baseline (mean)
  sig        : noise amplitute (standard deviation)
  c          : correlation coefficient ~[0, 1]
  bin_size   : bin size used for time series
  n_trials   : total simulation trials

  Returns:
  r          : output corr. coe.
  sp_rate    : spike rate
  sp1        : spike times of neuron 1 in the last trial
  sp2        : spike times of neuron 2 in the last trial
  """

    r12 = np.zeros(n_trials)
    sp_rate = np.zeros(n_trials)
    for i_trial in range(n_trials):
        I1gL, I2gL = correlate_input(pars, mu, sig, c)
        _, sp1 = run_LIF(pars, pars['g_L'] * I1gL)
        _, sp2 = run_LIF(pars, pars['g_L'] * I2gL)

        my_bin = np.arange(0, pars['T'], bin_size)

        sp1_count, _ = np.histogram(sp1, bins=my_bin)
        sp2_count, _ = np.histogram(sp2, bins=my_bin)

        r12[i_trial] = my_CC(sp1_count[::20], sp2_count[::20])
        sp_rate[i_trial] = len(sp1) / pars['T'] * 1000.

    return r12.mean(), sp_rate.mean(), sp1, sp2


# @title Plotting Functions

def example_plot_myCC(correlate_input=None, my_CC=None):
    pars = default_pars(T=50000, dt=.1)

    c = np.arange(10) * 0.1
    r12 = np.zeros(10)
    for i in range(10):
        I1gL, I2gL = correlate_input(pars, mu=20.0, sig=7.5, c=c[i])
        r12[i] = my_CC(I1gL, I2gL)

    plt.figure()
    plt.plot(c, r12, 'bo', alpha=0.7, label='Simulation', zorder=2)
    plt.plot([-0.05, 0.95], [-0.05, 0.95], 'k--', label='y=x',
             dashes=(2, 2), zorder=1)
    plt.xlabel('True CC')
    plt.ylabel('Sample CC')
    plt.legend(loc='best')
    plt.show()


def my_raster_Poisson(range_t, spike_train, n):
    """
  Ffunction generates and plots the raster of the Poisson spike train

  Args:
    range_t     : time sequence
    spike_train : binary spike trains, with shape (N, Lt)
    n           : number of Poisson trains plot

  Returns:
    Raster plot of the spike train
  """

    # find the number of all the spike trains
    N = spike_train.shape[0]

    # n should smaller than N:
    if n > N:
        print('The number n exceeds the size of spike trains')
        print('The number n is set to be the size of spike trains')
        n = N

    # plot rater
    plt.figure()
    i = 0
    while i < n:
        if spike_train[i, :].sum() > 0.:
            t_sp = range_t[spike_train[i, :] > 0.5]  # spike times
            plt.plot(t_sp, i * np.ones(len(t_sp)), 'k|', ms=10, markeredgewidth=2)
        i += 1
    plt.xlim([range_t[0], range_t[-1]])
    plt.ylim([-0.5, n + 0.5])
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Neuron ID', fontsize=12)
    plt.show()


def plot_c_r_LIF(c, r, mycolor, mylabel):
    z = np.polyfit(c, r, deg=1)
    c_range = np.array([c.min() - 0.05, c.max() + 0.05])
    plt.plot(c, r, 'o', color=mycolor, alpha=0.7, label=mylabel, zorder=2)
    plt.plot(c_range, z[0] * c_range + z[1], color=mycolor, zorder=1)


# @title Plotting Functions

def my_illus_LIFSYN(pars, v_fmp, v):
    """
  Illustartion of FMP and membrane voltage

  Args:
    pars  : parameters dictionary
    v_fmp : free membrane potential, mV
    v     : membrane voltage, mV

  Returns:
    plot of membrane voltage and FMP, alongside with the spiking threshold
    and the mean FMP (dashed lines)
  """

    plt.figure(figsize=(14, 5))
    plt.plot(pars['range_t'], v_fmp, 'r', lw=1.,
             label='Free mem. pot.', zorder=2)
    plt.plot(pars['range_t'], v, 'b', lw=1.,
             label='True mem. pot', zorder=1, alpha=0.7)
    plt.axhline(pars['V_th'], 0, 1, color='k', lw=2., ls='--',
                label='Spike Threshold', zorder=1)
    plt.axhline(np.mean(v_fmp), 0, 1, color='r', lw=2., ls='--',
                label='Mean Free Mem. Pot.', zorder=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    plt.legend(loc=[1.02, 0.68])
    plt.show()


def plot_volt_trace(pars, v, sp, show=True):
    """
  Plot trajetory of membrane potential for a single neuron

  Args:
    pars   : parameter dictionary
    v      : volt trajetory
    sp     : spike train

  Returns:
    figure of the membrane potential trajetory for a single neuron
  """

    V_th = pars['V_th']
    dt = pars['dt']
    if sp.size:
        sp_num = (sp / dt).astype(int) - 1
        v[sp_num] += 10

    plt.plot(pars['range_t'], v, 'b')
    plt.axhline(V_th, 0, 1, color='k', ls='--', lw=1.)
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    if show:
        plt.show()


def Poisson_generator(pars, rate, n, myseed=False):
    """
    Generates poisson trains

    Args:
    pars       : parameter dictionary
    rate       : noise amplitute [Hz]
    n          : number of Poisson trains
    myseed     : random seed. int or boolean

    Returns:
    pre_spike_train : spike train matrix, ith row represents whether
                      there is a spike in ith spike train over time
                      (1 if spike, 0 otherwise)
    """

    # Retrieve simulation parameters
    dt, range_t = pars['dt'], pars['range_t']
    Lt = range_t.size

    # set random seed
    if myseed:
        np.random.seed(seed=myseed)
    else:
        np.random.seed()

    # generate uniformly distributed random variables
    u_rand = np.random.rand(n, Lt)

    # generate Poisson train
    poisson_train = 1. * (u_rand < rate * (dt / 1000.))

    return poisson_train


def dynamic_syn(g_bar, tau_syn, U0, tau_d, tau_f, pre_spike_train, dt):
    """
    Short-term synaptic plasticity

    Args:
    g_bar           : synaptic conductance strength
    tau_syn         : synaptic time constant [ms]
    U0              : synaptic release probability at rest
    tau_d           : synaptic depression time constant of x [ms]
    tau_f           : synaptic facilitation time constantr of u [ms]
    pre_spike_train : total spike train (number) input
                      from presynaptic neuron
    dt              : time step [ms]

    Returns:
    u               : usage of releasable neurotransmitter
    R               : fraction of synaptic neurotransmitter resources available
    g               : postsynaptic conductance

    """

    Lt = len(pre_spike_train)
    # Initialize
    u = np.zeros(Lt)
    R = np.zeros(Lt)
    R[0] = 1.
    g = np.zeros(Lt)

    # simulation
    for it in range(Lt - 1):
        # Compute du
        du = -(dt / tau_f) * u[it] + U0 * (1.0 - u[it]) * pre_spike_train[it + 1]
        u[it + 1] = u[it] + du
        # Compute dR
        dR = (dt / tau_d) * (1.0 - R[it]) - u[it + 1] * R[it] * pre_spike_train[it + 1]
        R[it + 1] = R[it] + dR
        # Compute dg
        dg = -(dt / tau_syn) * g[it] + g_bar * R[it] * u[it + 1] * pre_spike_train[it + 1]
        g[it + 1] = g[it] + dg

    return u, R, g


def my_illus_STD(Poisson=False, rate=20., U0=0.5,
                 tau_d=100., tau_f=50., plot_out=True):
    """
   Only for one presynaptic train

  Args:
    Poisson    : Poisson or regular input spiking trains
    rate       : Rate of input spikes, Hz
    U0         : synaptic release probability at rest
    tau_d      : synaptic depression time constant of x [ms]
    tau_f      : synaptic facilitation time constantr of u [ms]
    plot_out   : whether ot not to plot, True or False

  Returns:
    Nothing.
  """

    T_simu = 10.0 * 1000 / (1.0 * rate)  # 10 spikes in the time window
    pars = default_pars(T=T_simu)
    dt = pars['dt']

    if Poisson:
        # Poisson type spike train
        pre_spike_train = Poisson_generator(pars, rate, n=1)
        pre_spike_train = pre_spike_train.sum(axis=0)
    else:
        # Regular firing rate
        isi_num = int((1e3 / rate) / dt)  # number of dt
        pre_spike_train = np.zeros(len(pars['range_t']))
        pre_spike_train[::isi_num] = 1.

    u, R, g = dynamic_syn(g_bar=1.2, tau_syn=5., U0=U0,
                          tau_d=tau_d, tau_f=tau_f,
                          pre_spike_train=pre_spike_train,
                          dt=pars['dt'])

    if plot_out:
        plt.figure(figsize=(12, 6))

        plt.subplot(221)
        plt.plot(pars['range_t'], R, 'b', label='R')
        plt.plot(pars['range_t'], u, 'r', label='u')
        plt.legend(loc='best')
        plt.xlim((0, pars['T']))
        plt.ylabel(r'$R$ or $u$ (a.u)')
        plt.subplot(223)
        spT = pre_spike_train > 0
        t_sp = pars['range_t'][spT]  # spike times
        plt.plot(t_sp, 0. * np.ones(len(t_sp)), 'k|', ms=18, markeredgewidth=2)
        plt.xlabel('Time (ms)');
        plt.xlim((0, pars['T']))
        plt.yticks([])
        plt.title('Presynaptic spikes')

        plt.subplot(122)
        plt.plot(pars['range_t'], g, 'r', label='STP synapse')
        plt.xlabel('Time (ms)')
        plt.ylabel('g (nS)')
        plt.xlim((0, pars['T']))

        plt.tight_layout()
        plt.show()

    if not Poisson:
        return g[isi_num], g[9 * isi_num]


# @title Plotting functions
def my_raster_plot(range_t, spike_train, n):
    """Generates poisson trains

  Args:
    range_t     : time sequence
    spike_train : binary spike trains, with shape (N, Lt)
    n           : number of Poisson trains plot

  Returns:
    Raster_plot of the spike train
  """

    # Find the number of all the spike trains
    N = spike_train.shape[0]

    # n should be smaller than N:
    if n > N:
        print('The number n exceeds the size of spike trains')
        print('The number n is set to be the size of spike trains')
        n = N

    # Raster plot
    i = 0
    while i <= n:
        if spike_train[i, :].sum() > 0.:
            t_sp = range_t[spike_train[i, :] > 0.5]  # spike times
            plt.plot(t_sp, i * np.ones(len(t_sp)), 'k|', ms=10, markeredgewidth=2)
        i += 1
    plt.xlim([range_t[0], range_t[-1]])
    plt.ylim([-0.5, n + 0.5])
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.show()


def my_example_P(pre_spike_train_ex, pars, P):
    """Generates two plots (raster plot and LTP vs time plot)

  Args:
    pre_spike_train_ex     : spike-train
    pars : dictionary with the parameters
    P : LTP ratio

  Returns:
    my_example_P returns a rastert plot (top),
    and a LTP ratio across time (bottom)
  """
    spT = pre_spike_train_ex
    plt.figure(figsize=(7, 6))
    plt.subplot(211)
    color_set = ['red', 'blue', 'black', 'orange', 'cyan']
    for i in range(spT.shape[0]):
        t_sp = pars['range_t'][spT[i, :] > 0.5]  # spike times
        plt.plot(t_sp, i * np.ones(len(t_sp)), '|',
                 color=color_set[i],
                 ms=10, markeredgewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.xlim(0, 200)

    plt.subplot(212)
    for k in range(5):
        plt.plot(pars['range_t'], P[k, :], color=color_set[k], lw=1.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('P(t)')
    plt.xlim(0, 200)

    plt.tight_layout()
    plt.show()


def mySTDP_plot(A_plus, A_minus, tau_stdp, time_diff, dW):
    plt.figure()
    plt.plot([-5 * tau_stdp, 5 * tau_stdp], [0, 0], 'k', linestyle=':')
    plt.plot([0, 0], [-A_minus, A_plus], 'k', linestyle=':')

    plt.plot(time_diff[time_diff <= 0], dW[time_diff <= 0], 'ro')
    plt.plot(time_diff[time_diff > 0], dW[time_diff > 0], 'bo')

    plt.xlabel(r't$_{\mathrm{pre}}$ - t$_{\mathrm{post}}$ (ms)')
    plt.ylabel(r'$\Delta$W', fontsize=12)
    plt.title('Biphasic STDP', fontsize=12, fontweight='bold')
    plt.show()

