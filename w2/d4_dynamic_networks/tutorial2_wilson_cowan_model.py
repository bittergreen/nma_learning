# Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt  # root-finding algorithm

from w2.d4_dynamic_networks.utils import default_pars, my_test_plot, F, plot_FI_inverse, get_E_nullcline, \
    get_I_nullcline, plot_nullclines, plot_complete_analysis


def simulate_wc(tau_E, a_E, theta_E, tau_I, a_I, theta_I,
                wEE, wEI, wIE, wII, I_ext_E, I_ext_I,
                rE_init, rI_init, dt, range_t, **other_pars):
    """
    Simulate the Wilson-Cowan equations

    Args:
    Parameters of the Wilson-Cowan model

    Returns:
    rE, rI (arrays) : Activity of excitatory and inhibitory populations
    """
    # Initialize activity arrays
    Lt = range_t.size
    rE = np.append(rE_init, np.zeros(Lt - 1))
    rI = np.append(rI_init, np.zeros(Lt - 1))
    I_ext_E = I_ext_E * np.ones(Lt)
    I_ext_I = I_ext_I * np.ones(Lt)

    # Simulate the Wilson-Cowan equations
    for k in range(Lt - 1):
        # Calculate the derivative of the E population
        drE = (dt / tau_E) * (-rE[k] + F(wEE * rE[k] - wEI * rI[k] + I_ext_E[k], a_E, theta_E))

        # Calculate the derivative of the I population
        drI = (dt / tau_I) * (-rI[k] + F(-wII * rI[k] + wIE * rE[k] + I_ext_I[k], a_I, theta_I))

        # Update using Euler's method
        rE[k + 1] = rE[k] + drE
        rI[k + 1] = rI[k] + drI

    return rE, rI


def section1_wc_model():
    pars = default_pars()

    # Simulate first trajectory
    rE1, rI1 = simulate_wc(**default_pars(rE_init=.32, rI_init=.15))

    # Simulate second trajectory
    rE2, rI2 = simulate_wc(**default_pars(rE_init=.33, rI_init=.15))

    # Visualize
    my_test_plot(pars['range_t'], rE1, rI1, rE2, rI2, pars)


def plot_activity_phase(n_t):
    pars = default_pars(T=10, rE_init=0.6, rI_init=0.8)
    rE, rI = simulate_wc(**pars)
    plt.figure(figsize=(8, 5.5))
    plt.subplot(211)
    plt.plot(pars['range_t'], rE, 'b', label=r'$r_E$')
    plt.plot(pars['range_t'], rI, 'r', label=r'$r_I$')
    plt.plot(pars['range_t'][n_t], rE[n_t], 'bo')
    plt.plot(pars['range_t'][n_t], rI[n_t], 'ro')
    plt.axvline(pars['range_t'][n_t], 0, 1, color='k', ls='--')
    plt.xlabel('t (ms)', fontsize=14)
    plt.ylabel('Activity', fontsize=14)
    plt.legend(loc='best', fontsize=14)

    plt.subplot(212)
    plt.plot(rE, rI, 'k')
    plt.plot(rE[n_t], rI[n_t], 'ko')
    plt.xlabel(r'$r_E$', fontsize=18, color='b')
    plt.ylabel(r'$r_I$', fontsize=18, color='r')

    plt.tight_layout()
    plt.show()


def section2_phase_plane_analysis():
    # plot_activity_phase(2)

    # Set parameters
    x = np.linspace(1e-6, 1, 100)

    # Get inverse and visualize
    # plot_FI_inverse(x, a=1, theta=3)

    # Set parameters
    pars = default_pars()
    Exc_null_rE = np.linspace(-0.01, 0.96, 100)
    Inh_null_rI = np.linspace(-.01, 0.8, 100)

    # Compute nullclines
    Exc_null_rI = get_E_nullcline(Exc_null_rE, **pars)
    Inh_null_rE = get_I_nullcline(Inh_null_rI, **pars)

    # Visualize
    # plot_nullclines(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI)

    # Create vector field using EIderivs
    plot_complete_analysis(pars)


if __name__ == '__main__':
    # section1_wc_model()
    section2_phase_plane_analysis()
