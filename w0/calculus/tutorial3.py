import math
import numpy as np

from utils import visualize_population_approx, plot_IF, plot_rErI, plot_rErI_Simple


def euler_1():
    dt = 0.5
    t = np.arange(1, 5+dt/2, dt)
    p = np.zeros_like(t)
    p[0] = math.e ** 0.3
    for i in range(len(p)-1):
        p[i+1] = p[i] + dt * 0.3 * p[i]
    visualize_population_approx(t, p)


def euler_lif(I, time, dt):
    V_reset = -75
    E_L = -75
    tau_m = 10
    R_m = 10
    V_th = -50

    t_isi = 0
    spike = 0
    spike_times = []
    v = np.ones_like(time) * V_reset

    for i in range(len(time) - 1):
        v[i+1] = v[i] + dt * ((-(v[i] - E_L) + R_m * I[i]) / tau_m)
        if v[i] > V_th:
            v[i] = 0
            v[i+1] = V_reset
            spike += 1
            spike_times.append(time[i])
            t_isi = time[i+1]

    return spike, spike_times, v


def run_euler_lif():
    dt = 1
    t = np.arange(0, 1000, dt)
    I = np.sin(4 * 2 * np.pi * t / 1000) + 2
    euler_lif(I, t, dt)
    Spike, Spike_time, V = euler_lif(I, t, dt)
    # Visualize
    plot_IF(t, V, I, Spike_time)


def euler_simple_linear_system(t, dt):
    # Set up parameters
    tau_E = 100
    tau_I = 120
    n = len(t)
    r_I = np.zeros(n)
    r_I[0] = 20
    r_E = np.zeros(n)
    r_E[0] = 30

    W_EI = 0.6
    W_IE = -5
    W_EE = 0.8
    W_II = -1

    # Loop over time steps
    for k in range(n - 1):
        # Estimate r_e
        dr_E = (W_IE * r_I[k] + W_EE * r_E[k]) / tau_E
        r_E[k + 1] = r_E[k] + dt * dr_E

        # Estimate r_i
        dr_I = (W_EI * r_E[k] + W_II * r_I[k]) / tau_I
        r_I[k + 1] = r_I[k] + dt * dr_I

    return r_E, r_I


def run_euler_simple_linear_system():
    # Set up dt, t
    dt = 0.1  # time-step
    t = np.arange(0, 1000, dt)

    # Run Euler method
    r_E, r_I = euler_simple_linear_system(t, dt)

    # Visualize
    plot_rErI_Simple(t, r_E, r_I)
    # plot_rErI(t, r_E, r_I)


if __name__ == '__main__':
    # euler_1()
    run_euler_simple_linear_system()

