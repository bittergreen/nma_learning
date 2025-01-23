# Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt  # root-finding algorithm

from w2.d4_dynamic_networks.utils import default_pars, get_E_nullcline, get_I_nullcline, plot_nullclines, EIderivs, F, \
    my_plot_nullcline, plot_fp, dF, simulate_wc, my_plot_trajectories, my_plot_vector


def my_fp(pars, rE_init, rI_init):
    """
    Use opt.root function to solve Equations (2)-(3) from initial values
    """

    tau_E, a_E, theta_E = pars['tau_E'], pars['a_E'], pars['theta_E']
    tau_I, a_I, theta_I = pars['tau_I'], pars['a_I'], pars['theta_I']
    wEE, wEI = pars['wEE'], pars['wEI']
    wIE, wII = pars['wIE'], pars['wII']
    I_ext_E, I_ext_I = pars['I_ext_E'], pars['I_ext_I']

    # define the right hand of wilson-cowan equations
    def my_WCr(x):
        rE, rI = x
        drEdt = (-rE + F(wEE * rE - wEI * rI + I_ext_E, a_E, theta_E)) / tau_E
        drIdt = (-rI + F(wIE * rE - wII * rI + I_ext_I, a_I, theta_I)) / tau_I
        y = np.array([drEdt, drIdt])

        return y

    x0 = np.array([rE_init, rI_init])
    x_fp = opt.root(my_WCr, x0).x

    return x_fp


def check_fp(pars, x_fp, mytol=1e-6):
    """
    Verify (drE/dt)^2 + (drI/dt)^2< mytol

    Args:
    pars    : Parameter dictionary
    fp      : value of fixed point
    mytol   : tolerance, default as 10^{-6}

    Returns :
    Whether it is a correct fixed point: True/False
    """

    drEdt, drIdt = EIderivs(x_fp[0], x_fp[1], **pars)

    return drEdt ** 2 + drIdt ** 2 < mytol


def get_eig_Jacobian(fp,
                     tau_E, a_E, theta_E, wEE, wEI, I_ext_E,
                     tau_I, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    """Compute eigenvalues of the Wilson-Cowan Jacobian matrix at fixed point."""
    # Initialization
    rE, rI = fp
    J = np.zeros((2, 2))

    # Compute the four elements of the Jacobian matrix
    J[0, 0] = (-1 + wEE * dF(wEE * rE - wEI * rI + I_ext_E,
                             a_E, theta_E)) / tau_E

    J[0, 1] = (-wEI * dF(wEE * rE - wEI * rI + I_ext_E,
                         a_E, theta_E)) / tau_E

    J[1, 0] = (wIE * dF(wIE * rE - wII * rI + I_ext_I,
                        a_I, theta_I)) / tau_I

    J[1, 1] = (-1 - wII * dF(wIE * rE - wII * rI + I_ext_I,
                             a_I, theta_I)) / tau_I

    # Compute and return the eigenvalues
    evals = np.linalg.eig(J)[0]
    return evals


def plot_nullcline_diffwEE(wEE):
    """
    plot nullclines for different values of wEE
    """

    pars = default_pars(wEE=wEE)

    # plot the E, I nullclines
    Exc_null_rE = np.linspace(-0.01, .96, 100)
    Exc_null_rI = get_E_nullcline(Exc_null_rE, **pars)

    Inh_null_rI = np.linspace(-.01, .8, 100)
    Inh_null_rE = get_I_nullcline(Inh_null_rI, **pars)

    plt.figure(figsize=(12, 5.5))
    plt.subplot(121)
    plt.plot(Exc_null_rE, Exc_null_rI, 'b', label='E nullcline')
    plt.plot(Inh_null_rE, Inh_null_rI, 'r', label='I nullcline')
    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')
    plt.legend(loc='best')

    plt.subplot(222)
    pars['rE_init'], pars['rI_init'] = 0.2, 0.2
    rE, rI = simulate_wc(**pars)
    plt.plot(pars['range_t'], rE, 'b', label='E population', clip_on=False)
    plt.plot(pars['range_t'], rI, 'r', label='I population', clip_on=False)
    plt.ylabel('Activity')
    plt.legend(loc='best')
    plt.ylim(-0.05, 1.05)
    plt.title('E/I activity\nfor different initial conditions',
              fontweight='bold')

    plt.subplot(224)
    pars['rE_init'], pars['rI_init'] = 0.4, 0.1
    rE, rI = simulate_wc(**pars)
    plt.plot(pars['range_t'], rE, 'b', label='E population', clip_on=False)
    plt.plot(pars['range_t'], rI, 'r', label='I population', clip_on=False)
    plt.xlabel('t (ms)')
    plt.ylabel('Activity')
    plt.legend(loc='best')
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()


def section1_stability_analysis():
    # Set parameters
    pars = default_pars()
    Exc_null_rE = np.linspace(-0.01, 0.96, 100)
    Inh_null_rI = np.linspace(-.01, 0.8, 100)

    # Compute nullclines
    Exc_null_rI = get_E_nullcline(Exc_null_rE, **pars)
    Inh_null_rE = get_I_nullcline(Inh_null_rI, **pars)

    pars = default_pars()

    my_plot_nullcline(pars)

    # Find the first fixed point
    x_fp_1 = my_fp(pars, 0.1, 0.1)
    if check_fp(pars, x_fp_1):
        plot_fp(x_fp_1)

    # Find the second fixed point
    x_fp_2 = my_fp(pars, 0.3, 0.3)
    if check_fp(pars, x_fp_2):
        plot_fp(x_fp_2)

    # Find the third fixed point
    x_fp_3 = my_fp(pars, 0.8, 0.6)
    if check_fp(pars, x_fp_3):
        plot_fp(x_fp_3)

    # Compute eigenvalues of Jacobian
    eig_1 = get_eig_Jacobian(x_fp_1, **pars)
    eig_2 = get_eig_Jacobian(x_fp_2, **pars)
    eig_3 = get_eig_Jacobian(x_fp_3, **pars)

    print(eig_1, 'Stable point')
    print(eig_2, 'Unstable point')
    print(eig_3, 'Stable point')

    plot_nullcline_diffwEE(6.0)


def time_constant_effect(tau_i=0.5):
    pars = default_pars(T=100.)
    pars['wEE'], pars['wEI'] = 6.4, 4.8
    pars['wIE'], pars['wII'] = 6.0, 1.2
    pars['I_ext_E'] = 0.8

    pars['tau_I'] = tau_i

    Exc_null_rE = np.linspace(0.0, .9, 100)
    Inh_null_rI = np.linspace(0.0, .6, 100)

    Exc_null_rI = get_E_nullcline(Exc_null_rE, **pars)
    Inh_null_rE = get_I_nullcline(Inh_null_rI, **pars)

    plt.figure(figsize=(12.5, 5.5))

    plt.subplot(121)  # nullclines
    plt.plot(Exc_null_rE, Exc_null_rI, 'b', label='E nullcline', zorder=2)
    plt.plot(Inh_null_rE, Inh_null_rI, 'r', label='I nullcline', zorder=2)
    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')

    # fixed point
    x_fp_1 = my_fp(pars, 0.5, 0.5)
    plt.plot(x_fp_1[0], x_fp_1[1], 'ko', zorder=2)

    eig_1 = get_eig_Jacobian(x_fp_1, **pars)

    # trajectories
    for ie in range(5):
        for ii in range(5):
            pars['rE_init'], pars['rI_init'] = 0.1 * ie, 0.1 * ii
            rE_tj, rI_tj = simulate_wc(**pars)
            plt.plot(rE_tj, rI_tj, 'k', alpha=0.3, zorder=1)

    # vector field
    EI_grid_E = np.linspace(0., 1.0, 20)
    EI_grid_I = np.linspace(0., 0.6, 20)
    rE, rI = np.meshgrid(EI_grid_E, EI_grid_I)
    drEdt, drIdt = EIderivs(rE, rI, **pars)
    n_skip = 2
    plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
               drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
               angles='xy', scale_units='xy', scale=10, facecolor='c')
    plt.title(r'$\tau_I=$' + '%.1f ms' % tau_i)

    plt.subplot(122)  # sample E/I trajectories
    pars['rE_init'], pars['rI_init'] = 0.25, 0.25
    rE, rI = simulate_wc(**pars)
    plt.plot(pars['range_t'], rE, 'b', label=r'$r_E$')
    plt.plot(pars['range_t'], rI, 'r', label=r'$r_I$')
    plt.xlabel('t (ms)')
    plt.ylabel('Activity')
    plt.title(r'$\tau_I=$' + '%.1f ms' % tau_i)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def section1_4():
    # @markdown Make sure you execute this cell to see the oscillations!

    pars = default_pars(T=100.)
    pars['wEE'], pars['wEI'] = 6.4, 4.8
    pars['wIE'], pars['wII'] = 6.0, 1.2
    pars['I_ext_E'] = 0.8
    pars['rE_init'], pars['rI_init'] = 0.25, 0.25

    rE, rI = simulate_wc(**pars)
    plt.figure(figsize=(8, 5.5))
    plt.plot(pars['range_t'], rE, 'b', label=r'$r_E$')
    plt.plot(pars['range_t'], rI, 'r', label=r'$r_I$')
    plt.xlabel('t (ms)')
    plt.ylabel('Activity')
    plt.legend(loc='best')
    plt.show()

    # @markdown Execute to visualize phase plane

    pars = default_pars(T=100.)
    pars['wEE'], pars['wEI'] = 6.4, 4.8
    pars['wIE'], pars['wII'] = 6.0, 1.2
    pars['I_ext_E'] = 0.8

    plt.figure(figsize=(7, 5.5))
    my_plot_nullcline(pars)

    # Find the correct fixed point
    x_fp_1 = my_fp(pars, 0.8, 0.8)
    if check_fp(pars, x_fp_1):
        plot_fp(x_fp_1, position=(0, 0), rotation=40)

    my_plot_trajectories(pars, 0.2, 3,
                         'Sample trajectories \nwith different initial values')

    my_plot_vector(pars)

    plt.legend(loc=[1.01, 0.7])
    plt.xlim(-0.05, 1.01)
    plt.ylim(-0.05, 0.65)
    plt.show()


if __name__ == '__main__':
    # section1_stability_analysis()
    # section1_4()
    time_constant_effect(1.5)
