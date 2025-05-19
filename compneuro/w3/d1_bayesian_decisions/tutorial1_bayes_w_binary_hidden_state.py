# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, transforms, gridspec
from scipy.optimize import fsolve
from collections import namedtuple

from utils import *


def U(s, a):
    # s - actual fact that fish is on the left or right
    # a - the action to fish on the left or right side
    if s == "left":
        if a == "left":
            value = 2
        else:
            value = -3
    else:
        if a == "left":
            value = -2
        else:
            value = 1
    return value


def utility_func(p_left):
    # calculating the utility of fishing on the left or right
    p_right = 1.0 - p_left
    u_left = U(s="left", a="left") * p_left + U(s="right", a="left") * p_right
    u_right = U(s="left", a="right") * p_left + U(s="right", a="right") * p_right
    print(f"When p_left is {p_left}, utility of fishing left is {u_left}, fishing right is {u_right}.")
    return u_left, u_right


def make_corr_plot(px, py, cor):
    Cmin, Cmax = compute_cor_range(px, py)  # allow correlation values
    P = compute_marginal(px, py, cor)
    fig = plot_joint_probs(P)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # utility_func(0.4)
    make_corr_plot(px=0.65, py=0.65, cor=0.8)

