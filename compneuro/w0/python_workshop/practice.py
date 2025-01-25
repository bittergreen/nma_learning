import math

import numpy as np
import matplotlib.pyplot as plt

t_max = 150e-3  # second
dt = 1e-3  # second
tau = 20e-3  # second
el = -60e-3  # milivolt
vr = -70e-3  # milivolt
vth = -50e-3  # milivolt
r = 100e6  # ohm
i_mean = 25e-11  # ampere


np.random.seed(2020)

step_end = int(t_max / dt)  # =150

if __name__ == '__main__':

    plt.figure()

    i_arr = np.arange(step_end, dtype=np.float64)

    for step in range(step_end):
        t = step * dt
        i = i_mean * (1 + math.sin(2 * math.pi * t / 0.01))
        i_arr[step] = i
        plt.plot(t, i, 'ko')
    print(i_arr)

    plt.show()



