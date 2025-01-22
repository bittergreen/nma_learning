# Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt  # root-finding algorithm


# @title Plotting Functions

def plot_fI(x, f):
    plt.figure(figsize=(6, 4))  # plot the figure
    plt.plot(x, f, 'k')
    plt.xlabel('x (a.u.)', fontsize=14)
    plt.ylabel('F(x)', fontsize=14)
    plt.show()


def plot_dr_r(r, drdt, x_fps=None):
    plt.figure()
    plt.plot(r, drdt, 'k')
    plt.plot(r, 0. * r, 'k--')
    if x_fps is not None:
        plt.plot(x_fps, np.zeros_like(x_fps), "ko", ms=12)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\frac{dr}{dt}$', fontsize=20)
    plt.ylim(-0.1, 0.1)
    plt.show()


def plot_dFdt(x, dFdt):
    plt.figure()
    plt.plot(x, dFdt, 'r')
    plt.xlabel('x (a.u.)', fontsize=14)
    plt.ylabel('dF(x)', fontsize=14)
    plt.show()
