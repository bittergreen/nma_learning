# Imports
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp  # numerical integration solver

from w2.d1_linear_systems.utils import plot_trajectory, plot_specific_example_stream_plots


def integrate_exponential(a, x0, dt, T):
    """Compute solution of the differential equation xdot=a*x with
  initial condition x0 for a duration T. Use time step dt for numerical
  solution.

  Args:
    a (scalar): parameter of xdot (xdot=a*x)
    x0 (scalar): initial condition (x at time 0)
    dt (scalar): timestep of the simulation
    T (scalar): total duration of the simulation

  Returns:
    ndarray, ndarray: `x` for all simulation steps and the time `t` at each step
  """

    # Initialize variables
    t = np.arange(0, T, dt)
    x = np.zeros_like(t, dtype=complex)
    x[0] = x0  # This is x at time t_0

    # Step through system and integrate in time
    for k in range(1, len(t)):
        # for each point in time, compute xdot from x[k-1]
        xdot = a * x[k - 1]

        # Update x based on x[k-1] and xdot
        x[k] = x[k - 1] + xdot * dt

    return x, t


def plot_euler_integration(real=-2, imaginary=-4):
    a = complex(real, imaginary)
    x, t = integrate_exponential(a, x0, dt, T)
    plt.plot(t, x.real)  # integrate exponential returns complex
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('x')
    plt.show()


def system(t, x, a00, a01, a10, a11):
    """
      Compute the derivative of the state x at time t for a linear
      differential equation with A matrix [[a00, a01], [a10, a11]].

      Args:
        t (float): time
        x (ndarray): state variable
        a00, a01, a10, a11 (float): parameters of the system

      Returns:
        ndarray: derivative xdot of state variable x at time t
    """
    # compute x1dot and x2dot
    x1dot = a00 * x[0] + a01 * x[1]
    x2dot = a10 * x[0] + a11 * x[1]

    return np.array([x1dot, x2dot])


if __name__ == '__main__':
    # Choose parameters
    a = -0.5  # parameter in f(x)
    T = 5  # total Time duration
    dt = 0.001  # timestep of our simulation
    x0 = 1.  # initial condition of x at time 0

    # plot_euler_integration(0, math.pi)

    # Set parameters
    T = 6  # total time duration
    dt = 0.1  # timestep of our simulation
    A = np.array([[2, -5],
                  [1, -2]])
    A = np.array([[3, 4],
                  [1, 2]])
    A = np.array([[-1, -1],
                  [0, 0.25]])
    x0 = [-0.1, 0.2]
    x0 = [10, 10]

    # Simulate and plot trajectories
    plot_trajectory(system, [A[0, 0], A[0, 1], A[1, 0], A[1, 1]], x0, dt=dt, T=T)

    A_option_1 = np.array([[2, -5], [1, -2]])
    A_option_2 = np.array([[3, 4], [1, 2]])
    A_option_3 = np.array([[-1, -1], [0, -0.25]])
    A_option_4 = np.array([[3, -2], [2, -2]])

    A_options = [A_option_1, A_option_2, A_option_3, A_option_4]
    plot_specific_example_stream_plots(A_options)
