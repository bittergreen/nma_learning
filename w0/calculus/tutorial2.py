import numpy as np
import sympy as sp
import IPython.display as ipd
from matplotlib import pyplot as plt

from utils import plot_dPdt, plot_V_no_input, plot_dVdt, Exact_Integrate_and_Fire, plot_IF


def shit1():
    plot_dPdt(alpha=0)
    plt.show()


def V_reset_widget(V_reset):
    plot_V_no_input(V_reset)


def Pop_widget(I):
    plot_dVdt(I=I)
    plt.show()


def shit2():
    dt = 0.5
    t_rest = 0

    t = np.arange(0, 1000, dt)

    tau_m = 10
    R_m = 10
    V_reset = E_L = -75

    I = 10

    V = E_L + R_m * I + (V_reset - E_L - R_m * I) * np.exp(-(t) / tau_m)

    with plt.xkcd():
        fig = plt.figure(figsize=(6, 4))
        plt.plot(t, V)
        plt.ylabel('V (mV)')
        plt.xlabel('time (ms)')
        plt.show()


def lif():
    # @markdown Make sure you execute this cell to be able to hear the neuron
    I = 2.7
    dt = 0.5
    t = np.arange(0, 1000, dt)
    Spike, Spike_time, V = Exact_Integrate_and_Fire(I, t)

    plot_IF(t, V, I, Spike_time)
    ipd.Audio(V, rate=len(V))


def my_lif():
    t, E_L, R_m, I, tao_m, V_reset = sp.symbols('t E_L R_m I tao_m V_reset')
    V_t = sp.Function('V')(t)
    dV_dt = sp.Derivative(V_t, t)

    equation = sp.Eq(dV_dt, (-(V_t - E_L) + R_m * I) / tao_m)

    # Solve the differential equation
    solution = sp.dsolve(equation, V_t, ics={V_t.subs(t, 0): V_reset})

    # Display the solution
    print(solution)


def F_I_curve():
    # @markdown *Execture this cell to visualize the FI curve*
    I_range = np.arange(2.0, 4.0, 0.1)
    dt = 0.5
    t = np.arange(0, 1000, dt)
    Spike_rate = np.ones(len(I_range))

    for i, I in enumerate(I_range):
        Spike_rate[i], _, _ = Exact_Integrate_and_Fire(I, t)

    with plt.xkcd():
        fig = plt.figure(figsize=(6, 4))
        plt.plot(I_range, Spike_rate)
        plt.xlabel('Input Current (nA)')
        plt.ylabel('Spikes per Second (Hz)')
        plt.show()


if __name__ == '__main__':
    # V_reset_widget(-100)
    # Pop_widget(-2)
    # shit2()
    # my_lif()
    # lif()
    F_I_curve()




