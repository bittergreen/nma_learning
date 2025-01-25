import numpy as np
from matplotlib import pyplot as plt, gridspec


def plot_Stuart_Landau(t, x, y, s):
    """
    Args:
      t  : time
      x  : x
      y  : y
      s : input
    Returns:
      figure with two panels
      top panel: Input as a function of time
      bottom panel: x
  """

    with plt.xkcd():
        fig = plt.figure(figsize=(14, 4))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1])

        # PLOT OF INPUT
        plt.subplot(gs[0])
        plt.ylabel(r'$s$')
        plt.plot(t, s, 'g')
        # plt.ylim((2,4))

        # PLOT OF ACTIVITY
        plt.subplot(gs[2])
        plt.plot(t, x)
        plt.ylabel(r'x')
        plt.xlabel(r't')
        plt.subplot(gs[3])
        plt.plot(x, y)
        plt.plot(x[0], y[0], 'go')
        plt.xlabel(r'x')
        plt.ylabel(r'y')
        plt.show()


def Euler_Stuart_Landau(s, time, dt, lamba=0.1, gamma=1.0, k=25):
    """
  Args:
    I: Input
    time: time
    dt: time-step
  """

    n = len(time)
    omega = 4 * 2 * np.pi
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = 1
    y[0] = 1

    for i in range(n - 1):
        dx = lamba * x[i] - omega * y[i] - gamma * (x[i] * x[i] + y[i] * y[i]) * x[i] + k * s[i]
        x[i + 1] = x[i] + dt * dx
        dy = lamba * y[i] + omega * x[i] - gamma * (x[i] * x[i] + y[i] * y[i]) * y[i]
        y[i + 1] = y[i] + dt * dy

    return x, y


def run_with_stable_input(lamda):
    dt = 0.1 / 1000
    t = np.arange(0, 3, dt)
    s = np.zeros(len(t))
    x, y = Euler_Stuart_Landau(s, t, dt, lamda)
    plot_Stuart_Landau(t, x, y, s)
    plt.show()


def run_with_sigmoid_input():
    dt = 0.1 / 1000
    t = np.arange(0, 3, dt)
    freq = 15.0
    s = np.sin(freq * 2 * np.pi * t)
    x, y = Euler_Stuart_Landau(s, t, dt, lamba=1, gamma=.1, k=50)
    plot_Stuart_Landau(t, x, y, s)


if __name__ == '__main__':
    run_with_sigmoid_input()

