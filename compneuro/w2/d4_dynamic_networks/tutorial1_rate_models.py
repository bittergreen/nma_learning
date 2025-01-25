# Imports

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt  # root-finding algorithm

from compneuro.w2.d4_dynamic_networks.utils import plot_fI, plot_dr_r


def default_pars_single(**kwargs):
    pars = {}

    # Excitatory parameters
    pars['tau'] = 1.  # Timescale of the E population [ms]
    pars['a'] = 1.2  # Gain of the E population
    pars['theta'] = 2.8  # Threshold of the E population

    # Connection strength
    pars['w'] = 0.  # E to E, we first set it to 0

    # External input
    pars['I_ext'] = 0.

    # simulation parameters
    pars['T'] = 20.  # Total duration of simulation [ms]
    pars['dt'] = .1  # Simulation time step [ms]
    pars['r_init'] = 0.2  # Initial value of E

    # External parameters if any
    pars.update(kwargs)

    # Vector of discretized time points [ms]
    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])

    return pars


def F(x, a, theta):
    """
    Population activation function.

    Args:
    x (float): the population input
    a (float): the gain of the function
    theta (float): the threshold of the function

    Returns:
    float: the population activation response F(x) for input x
    """

    # Define the sigmoidal transfer function f = F(x)
    f = (1 + np.exp(-a * (x - theta))) ** -1 - (1 + np.exp(a * theta)) ** -1

    return f


def section1_fi_curve():
    # Set parameters
    pars = default_pars_single()  # get default parameters
    x = np.arange(0, 10, .1)  # set the range of input

    # Compute transfer function
    f = F(x, pars['a'], pars['theta'])

    # Visualize
    plot_fI(x, f)


def simulate_single(pars):
    """
    Simulate an excitatory population of neurons

    Args:
    pars   : Parameter dictionary

    Returns:
    rE     : Activity of excitatory population (array)

    Example:
    pars = default_pars_single()
    r = simulate_single(pars)
    """

    # Set parameters
    tau, a, theta = pars['tau'], pars['a'], pars['theta']
    w = pars['w']
    I_ext = pars['I_ext']
    r_init = pars['r_init']
    dt, range_t = pars['dt'], pars['range_t']
    Lt = range_t.size

    # Initialize activity
    r = np.zeros(Lt)
    r[0] = r_init
    I_ext = I_ext * np.ones(Lt)

    # Update the E activity
    for k in range(Lt - 1):
        dr = dt / tau * (-r[k] + F(w * r[k] + I_ext[k], a, theta))
        r[k + 1] = r[k] + dr

    return r


def Myplot_E_diffI_difftau(I_ext, tau):
    # set external input and time constant
    pars = default_pars_single()
    pars['I_ext'] = I_ext
    pars['tau'] = tau

    # simulation
    r = simulate_single(pars)

    # Analytical Solution
    r_ana = (pars['r_init']
             + (F(I_ext, pars['a'], pars['theta'])
                - pars['r_init']) * (1. - np.exp(-pars['range_t'] / pars['tau'])))

    # plot
    plt.figure()
    plt.plot(pars['range_t'], r, 'b', label=r'$r_{\mathrm{sim}}$(t)', alpha=0.5,
             zorder=1)
    plt.plot(pars['range_t'], r_ana, 'b--', lw=5, dashes=(2, 2),
             label=r'$r_{\mathrm{ana}}$(t)', zorder=2)
    plt.plot(pars['range_t'],
             F(I_ext, pars['a'], pars['theta']) * np.ones(pars['range_t'].size),
             'k--', label=r'$F(I_{\mathrm{ext}})$')
    plt.xlabel('t (ms)', fontsize=16.)
    plt.ylabel('Activity r(t)', fontsize=16.)
    plt.legend(loc='best', fontsize=14.)
    plt.show()


def compute_drdt(r, I_ext, w, a, theta, tau, **other_pars):
    """Given parameters, compute dr/dt as a function of r.

    Args:
    r (1D array) : Average firing rate of the excitatory population
    I_ext, w, a, theta, tau (numbers): Simulation parameters to use
    other_pars : Other simulation parameters are unused by this function

    Returns
    drdt function for each value of r
    """
    # Calculate drdt
    drdt = (-r + F(w * r + I_ext, a, theta)) / tau

    return drdt


def my_fp_single(r_guess, a, theta, w, I_ext, **other_pars):
    """
    Calculate the fixed point through drE/dt=0

    Args:
    r_guess  : Initial value used for scipy.optimize function
    a, theta, w, I_ext : simulation parameters

    Returns:
    x_fp    : value of fixed point
    """

    # define the right hand of E dynamics
    def my_WCr(x):
        r = x
        drdt = (-r + F(w * r + I_ext, a, theta))
        y = np.array(drdt)

        return y

    x0 = np.array(r_guess)
    x_fp = opt.root(my_WCr, x0).x.item()

    return x_fp


def check_fp_single(x_fp, a, theta, w, I_ext, mytol=1e-4, **other_pars):
    """
    Verify |dr/dt| < mytol

    Args:
    fp      : value of fixed point
    a, theta, w, I_ext: simulation parameters
    mytol   : tolerance, default as 10^{-4}

    Returns :
    Whether it is a correct fixed point: True/False
    """
    # calculate Equation(3)
    y = x_fp - F(w * x_fp + I_ext, a, theta)

    # Here we set tolerance as 10^{-4}
    return np.abs(y) < mytol


def my_fp_finder(pars, r_guess_vector, mytol=1e-4):
    """
    Calculate the fixed point(s) through drE/dt=0

    Args:
    pars    : Parameter dictionary
    r_guess_vector  : Initial values used for scipy.optimize function
    mytol   : tolerance for checking fixed point, default as 10^{-4}

    Returns:
    x_fps   : values of fixed points

    """
    x_fps = []
    correct_fps = []
    for r_guess in r_guess_vector:
        x_fp = my_fp_single(r_guess, **pars)
        if check_fp_single(x_fp, **pars, mytol=mytol):
            x_fps.append(x_fp)

    return x_fps


# @markdown Execute to visualize dr/dt

def plot_intersection_single(w, I_ext):
    # set your parameters
    pars = default_pars_single(w=w, I_ext=I_ext)

    # find fixed points
    r_init_vector = [0, .4, .9]
    x_fps = my_fp_finder(pars, r_init_vector)
    print(f'Our fixed points are {x_fps}')

    # plot
    r = np.linspace(0, 1., 1000)
    drdt = (-r + F(w * r + I_ext, pars['a'], pars['theta'])) / pars['tau']

    plot_dr_r(r, drdt, x_fps)


def plot_single_diffEinit(r_init, w, I_ext):
    pars = default_pars_single(r_init=r_init, w=w, I_ext=I_ext)

    r = simulate_single(pars)

    plt.figure()
    plt.plot(pars['range_t'], r, 'b', zorder=1)
    plt.plot(0, r[0], 'bo', alpha=0.7, zorder=2)
    plt.xlabel('t (ms)', fontsize=16)
    plt.ylabel(r'$r(t)$', fontsize=16)
    plt.ylim(0, 1.0)
    plt.show()


def section2_fixed_points():
    plot_intersection_single(w=5.0, I_ext=0.5)
    # plot_single_diffEinit(0.45, 5.0, 0.5)

    pars = default_pars_single()
    pars['w'] = 5.0
    pars['I_ext'] = 0.5

    plt.figure(figsize=(8, 5))
    for ie in range(10):
        pars['r_init'] = 0.1 * ie  # set the initial value
        r = simulate_single(pars)  # run the simulation

        # plot the activity with given initial
        plt.plot(pars['range_t'], r, 'b', alpha=0.1 + 0.1 * ie,
                 label=r'r$_{\mathrm{init}}$=%.1f' % (0.1 * ie))

    plt.xlabel('t (ms)')
    plt.title('Two steady states?')
    plt.ylabel(r'$r$(t)')
    plt.legend(loc=[1.01, -0.06], fontsize=14)
    plt.show()


if __name__ == '__main__':
    # section1_fi_curve()
    # Myplot_E_diffI_difftau(I_ext=5.0, tau=3.0)
    section2_fixed_points()
