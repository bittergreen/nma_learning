# Imports
import matplotlib.pyplot as plt
import numpy as np
import time

from w2.d3_biological_neuron_models.utils import my_GWN, example_plot_myCC, default_pars, my_raster_Poisson, \
    LIF_output_cc, plot_c_r_LIF


# @markdown Execute this cell to get a function `correlate_input` for generating correlated GWN inputs
def correlate_input(pars, mu=20., sig=7.5, c=0.3):
    """
    Args:
    pars       : parameter dictionary
    mu         : noise baseline (mean)
    sig        : noise amplitute (standard deviation)
    c.         : correlation coefficient ~[0, 1]

    Returns:
    I1gL, I2gL : two correlated inputs with corr. coe. c
    """

    # generate Gaussian white noise xi_1, xi_2, xi_c
    xi_1 = my_GWN(pars, sig)
    xi_2 = my_GWN(pars, sig)
    xi_c = my_GWN(pars, sig)

    # Generate two correlated inputs by Equation. (1)
    I1gL = mu + np.sqrt(1. - c) * xi_1 + np.sqrt(c) * xi_c
    I2gL = mu + np.sqrt(1. - c) * xi_2 + np.sqrt(c) * xi_c

    return I1gL, I2gL


def my_CC(i, j):
    """
    Args:
    i, j  : two time series with the same length

    Returns:
    rij   : correlation coefficient
    """

    # Calculate the covariance of i and j
    cov = ((i - i.mean()) * (j - j.mean())).sum()

    # Calculate the variance of i
    var_i = ((i - i.mean()) * (i - i.mean())).sum()

    # Calculate the variance of j
    var_j = ((j - j.mean()) * (j - j.mean())).sum()

    # Calculate the correlation coefficient
    rij = cov / np.sqrt(var_i * var_j)

    return rij


# @markdown Execute this cell to get helper function `Poisson_generator`
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


# @markdown Execute this cell to get a function for generating correlated Poisson inputs (`generate_corr_Poisson`)


def generate_corr_Poisson(pars, poi_rate, c, myseed=False):
    """
    function to generate correlated Poisson type spike trains
    Args:
    pars       : parameter dictionary
    poi_rate   : rate of the Poisson train
    c.         : correlation coefficient ~[0, 1]

    Returns:
    sp1, sp2   : two correlated spike time trains with corr. coe. c
    """

    range_t = pars['range_t']

    mother_rate = poi_rate / c
    mother_spike_train = Poisson_generator(pars, rate=mother_rate,
                                           n=1, myseed=myseed)[0]
    sp_mother = range_t[mother_spike_train > 0]

    L_sp_mother = len(sp_mother)
    sp_mother_id = np.arange(L_sp_mother)
    L_sp_corr = int(L_sp_mother * c)

    np.random.shuffle(sp_mother_id)
    sp1 = np.sort(sp_mother[sp_mother_id[:L_sp_corr]])

    np.random.shuffle(sp_mother_id)
    sp2 = np.sort(sp_mother[sp_mother_id[:L_sp_corr]])

    return sp1, sp2


def corr_coeff_pairs(pars, rate, c, trials, bins):
    """
    Calculate the correlation coefficient of two spike trains, for different
    realizations

    Args:
      pars   : parameter dictionary
      rate   : rate of poisson inputs
      c      : correlation coefficient ~ [0, 1]
      trials : number of realizations
      bins   : vector with bins for time discretization

    Returns:
    r12      : correlation coefficient of a pair of inputs
    """

    r12 = np.zeros(trials)

    for i in range(trials):

        # Generate correlated Poisson inputs
        sp1, sp2 = generate_corr_Poisson(pars, rate, c, myseed=2020 + i)

        # Bin the spike times of the first input
        sp1_count, _ = np.histogram(sp1, bins=bins)

        # Bin the spike times of the second input
        sp2_count, _ = np.histogram(sp2, bins=bins)

        # Calculate the correlation coefficient
        r12[i] = my_CC(sp1_count, sp2_count)

    return r12


def measure_corr():
    poi_rate = 20.
    c = 0.2  # set true correlation
    pars = default_pars(T=10000)

    # bin the spike time
    bin_size = 20  # [ms]
    my_bin = np.arange(0, pars['T'], bin_size)
    n_trials = 100  # 100 realizations

    r12 = corr_coeff_pairs(pars, rate=poi_rate, c=c, trials=n_trials, bins=my_bin)
    print(f'True corr coe = {c:.3f}')
    print(f'Simu corr coe = {r12.mean():.3f}')


def section1_measurements():
    example_plot_myCC(correlate_input, my_CC)
    pars = default_pars()
    pre_spike_train = Poisson_generator(pars, rate=10, n=100, myseed=2020)
    my_raster_Poisson(pars['range_t'], pre_spike_train, 100)
    measure_corr()


def section2_i2o_correlation():
    # Play around with these parameters

    pars = default_pars(T=80000, dt=1.)  # get the parameters
    c_in = 0.3  # set input correlation value
    gwn_mean = 10.
    gwn_std = 10.

    # @markdown Do not forget to execute this cell to simulate the LIF

    bin_size = 10.  # ms
    starttime = time.perf_counter()  # time clock
    r12_ss, sp_ss, sp1, sp2 = LIF_output_cc(pars, mu=gwn_mean, sig=gwn_std, c=c_in,
                                            bin_size=bin_size, n_trials=10,
                                            correlate_input=correlate_input, my_CC=my_CC)

    # just the time counter
    endtime = time.perf_counter()
    timecost = (endtime - starttime) / 60.
    print(f"Simulation time = {timecost:.2f} min")

    print(f"Input correlation = {c_in}")
    print(f"Output correlation = {r12_ss}")

    plt.figure(figsize=(12, 6))
    plt.plot(sp1, np.ones(len(sp1)) * 1, '|', ms=20, label='neuron 1')
    plt.plot(sp2, np.ones(len(sp2)) * 1.1, '|', ms=20, label='neuron 2')
    plt.xlabel('time (ms)')
    plt.ylabel('neuron id.')
    plt.xlim(1000, 8000)
    plt.ylim(0.9, 1.2)
    plt.legend()
    plt.show()

    # @markdown Don't forget to execute this cell!

    pars = default_pars(T=80000, dt=1.)  # get the parameters
    bin_size = 10.
    c_in = np.arange(0, 1.0, 0.1)  # set the range for input CC
    r12_ss = np.zeros(len(c_in))  # small mu, small sigma

    starttime = time.perf_counter()  # time clock
    for ic in range(len(c_in)):
        r12_ss[ic], sp_ss, sp1, sp2 = LIF_output_cc(pars, mu=10.0, sig=10.,
                                                    c=c_in[ic], bin_size=bin_size,
                                                    n_trials=10, correlate_input=correlate_input, my_CC=my_CC)

    endtime = time.perf_counter()
    timecost = (endtime - starttime) / 60.
    print(f"Simulation time = {timecost:.2f} min")

    plot_c_r_LIF(c_in, r12_ss, mycolor='b', mylabel='Output CC')
    plt.plot([c_in.min() - 0.05, c_in.max() + 0.05],
             [c_in.min() - 0.05, c_in.max() + 0.05],
             'k--', dashes=(2, 2), label='y=x')

    plt.xlabel('Input CC')
    plt.ylabel('Output CC')
    plt.legend(loc='best', fontsize=16)
    plt.show()


def section3_correlation_transfer_function():
    # @markdown Execute this cell to visualize correlation transfer functions

    pars = default_pars(T=80000, dt=1.)  # get the parameters
    n_trials = 10
    bin_size = 10.
    c_in = np.arange(0., 1., 0.2)  # set the range for input CC
    r12_ss = np.zeros(len(c_in))  # small mu, small sigma
    r12_ls = np.zeros(len(c_in))  # large mu, small sigma
    r12_sl = np.zeros(len(c_in))  # small mu, large sigma

    starttime = time.perf_counter()  # time clock
    for ic in range(len(c_in)):
        r12_ss[ic], sp_ss, sp1, sp2 = LIF_output_cc(pars, mu=10.0, sig=10.,
                                                    c=c_in[ic], bin_size=bin_size,
                                                    n_trials=n_trials, correlate_input=correlate_input, my_CC=my_CC)
        r12_ls[ic], sp_ls, sp1, sp2 = LIF_output_cc(pars, mu=18.0, sig=10.,
                                                    c=c_in[ic], bin_size=bin_size,
                                                    n_trials=n_trials, correlate_input=correlate_input, my_CC=my_CC)
        r12_sl[ic], sp_sl, sp1, sp2 = LIF_output_cc(pars, mu=10.0, sig=20.,
                                                    c=c_in[ic], bin_size=bin_size,
                                                    n_trials=n_trials, correlate_input=correlate_input, my_CC=my_CC)
    endtime = time.perf_counter()
    timecost = (endtime - starttime) / 60.
    print(f"Simulation time = {timecost:.2f} min")

    plot_c_r_LIF(c_in, r12_ss, mycolor='b', mylabel=r'Small $\mu$, small $\sigma$')
    plot_c_r_LIF(c_in, r12_ls, mycolor='y', mylabel=r'Large $\mu$, small $\sigma$')
    plot_c_r_LIF(c_in, r12_sl, mycolor='r', mylabel=r'Small $\mu$, large $\sigma$')
    plt.plot([c_in.min() - 0.05, c_in.max() + 0.05],
             [c_in.min() - 0.05, c_in.max() + 0.05],
             'k--', dashes=(2, 2), label='y=x')
    plt.xlabel('Input CC')
    plt.ylabel('Output CC')
    plt.legend(loc='best', fontsize=14)
    plt.show()


if __name__ == '__main__':
    # section1_measurements()
    # section2_i2o_correlation()
    section3_correlation_transfer_function()
