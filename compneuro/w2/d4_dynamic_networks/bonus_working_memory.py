# Imports
import matplotlib.pyplot as plt
import numpy as np

from compneuro.w2.d4_dynamic_networks.utils import default_pars, simulate_wc


def my_OU(pars, sig, myseed=False):
    """
    Expects:
    pars       : parameter dictionary
    sig        : noise amplitute
    myseed     : random seed. int or boolean

    Returns:
    I          : Ornstein-Uhlenbeck input current
    """

    # Retrieve simulation parameters
    dt, range_t = pars['dt'], pars['range_t']
    Lt = range_t.size
    tau_ou = pars['tau_ou']  # [ms]

    # set random seed
    if myseed:
        np.random.seed(seed=myseed)
    else:
        np.random.seed()

    # Initialize
    noise = np.random.randn(Lt)
    I_ou = np.zeros(Lt)
    I_ou[0] = noise[0] * sig

    # generate OU
    for it in range(Lt - 1):
        I_ou[it + 1] = (I_ou[it]
                        + dt / tau_ou * (0. - I_ou[it])
                        + np.sqrt(2 * dt / tau_ou) * sig * noise[it + 1])
    return I_ou


def simulate_ou():
    pars = default_pars(T=50)
    pars['tau_ou'] = 1.  # [ms]
    sig_ou = 0.1
    I_ou = my_OU(pars, sig=sig_ou, myseed=2020)
    plt.figure(figsize=(8, 5.5))
    plt.plot(pars['range_t'], I_ou, 'b')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$I_{\mathrm{OU}}$')
    plt.show()


def wc_with_ou_input():
    # @markdown Execute this cell to plot activity with noisy input current
    pars = default_pars(T=100)
    pars['tau_ou'] = 1.  # [ms]
    sig_ou = 0.1
    pars['I_ext_E'] = my_OU(pars, sig=sig_ou, myseed=20201)
    pars['I_ext_I'] = my_OU(pars, sig=sig_ou, myseed=20202)

    pars['rE_init'], pars['rI_init'] = 0.1, 0.1
    rE, rI = simulate_wc(**pars)

    plt.figure(figsize=(8, 5.5))
    ax = plt.subplot(111)
    ax.plot(pars['range_t'], rE, 'b', label='E population')
    ax.plot(pars['range_t'], rI, 'r', label='I population')
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('Activity')
    ax.legend(loc='best')
    plt.show()


# @markdown Make sure you execute this cell to enable the widget!
def my_inject(pars, t_start, t_lag=10.):
    """
    Expects:
    pars       : parameter dictionary
    t_start    : pulse starts [ms]
    t_lag      : pulse lasts  [ms]

    Returns:
    I          : extra pulse time
    """

    # Retrieve simulation parameters
    dt, range_t = pars['dt'], pars['range_t']
    Lt = range_t.size

    # Initialize
    I = np.zeros(Lt)

    # pulse timing
    N_start = int(t_start / dt)
    N_lag = int(t_lag / dt)
    I[N_start:N_start + N_lag] = 1.

    return I


def WC_with_pulse(SE=0.):
    pars = default_pars(T=100)
    pars['tau_ou'] = 1.  # [ms]
    sig_ou = 0.1
    pars['I_ext_I'] = my_OU(pars, sig=sig_ou, myseed=2021)
    pars['rE_init'], pars['rI_init'] = 0.1, 0.1

    # pulse
    I_pulse = my_inject(pars, t_start=20., t_lag=10.)
    L_pulse = sum(I_pulse > 0.)

    pars['I_ext_E'] = my_OU(pars, sig=sig_ou, myseed=2022)
    pars['I_ext_E'] += SE * I_pulse

    rE, rI = simulate_wc(**pars)

    plt.figure(figsize=(8, 5.5))
    ax = plt.subplot(111)
    ax.plot(pars['range_t'], rE, 'b', label='E population')
    ax.plot(pars['range_t'], rI, 'r', label='I population')

    ax.plot(pars['range_t'][I_pulse > 0.], 1.0 * np.ones(L_pulse), 'r', lw=3.)
    ax.text(25, 1.05, 'stimulus on', horizontalalignment='center',
            verticalalignment='bottom')
    ax.set_ylim(-0.03, 1.2)
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('Activity')
    ax.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # simulate_ou()
    # wc_with_ou_input()
    WC_with_pulse(1.5)
