import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

if __name__ == '__main__':
    a = np.random.poisson(5, 1000)
    plt.hist(a, 20)
    plt.show()