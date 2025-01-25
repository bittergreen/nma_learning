import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def gen_poisson(lam, size):
    arr1 = np.random.poisson(lam, size)
    arr2 = stats.poisson(lam).rvs(size)
    print(arr1)
    print(arr2)
    plt.figure(figsize=(10, 6))
    plt.hist(arr1, bins=20, label='Numpy', alpha=0.3, color='green', edgecolor='black')
    plt.hist(arr2, bins=20, label='Scipy', alpha=0.3, color='orange', edgecolor='black')
    plt.title(f'Poisson Distribution (λ={lam}) - NumPy vs SciPy')
    plt.xlabel('Number of events')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    gen_poisson(10, 100)
