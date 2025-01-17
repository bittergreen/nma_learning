# Imports
import numpy as np
import matplotlib.pyplot as plt

from w2.d2_linear_systems.utils import plot_switch_simulation, plot_interswitch_interval_histogram, \
    plot_state_probabilities


# simulate state of our ion channel in time
# the two parameters that govern transitions are
# c2o: closed to open rate
# o2c: open to closed rate
def ion_channel_opening(c2o, o2c, T, dt):
    # initialize variables
    t = np.arange(0, T, dt)
    x = np.zeros_like(t)
    switch_times = []

    # assume we always start in Closed state
    x[0] = 0

    # generate a bunch of random uniformly distributed numbers
    # between zero and unity: [0, 1),
    # one for each dt in our simulation.
    # we will use these random numbers to model the
    # closed/open transitions
    myrand = np.random.random_sample(size=len(t))

    # walk through time steps of the simulation
    for k in range(len(t) - 1):
        # switching between closed/open states are
        # Poisson processes
        if x[k] == 0 and myrand[k] < c2o * dt:  # remember to scale by dt!
            x[k + 1:] = 1
            switch_times.append(k * dt)
        elif x[k] == 1 and myrand[k] < o2c * dt:
            x[k + 1:] = 0
            switch_times.append(k * dt)

    return t, x, switch_times


def section1_telegraph_process():
    # parameters
    T = 5000  # total Time duration
    dt = 0.001  # timestep of our simulation

    c2o = 0.02
    o2c = 0.1
    np.random.seed(0)  # set random seed
    t, x, switch_times = ion_channel_opening(c2o, o2c, T, .1)
    plot_switch_simulation(t, x)

    # hint: see np.diff()
    inter_switch_intervals = np.diff(switch_times)

    # plot inter-switch intervals
    plot_interswitch_interval_histogram(inter_switch_intervals)

    # @markdown Execute cell to visualize distribution of time spent in each state.

    states = ['Closed', 'Open']
    (unique, counts) = np.unique(x, return_counts=True)

    plt.figure()
    plt.bar(states, counts)
    plt.ylabel('Number of time steps')
    plt.xlabel('State of ion channel')
    plt.show()

    # @markdown Execute to visualize cumulative mean of state
    plt.figure()
    plt.plot(t, np.cumsum(x) / np.arange(1, len(t) + 1))
    plt.xlabel('time')
    plt.ylabel('Cumulative mean of state')
    plt.show()


def simulate_prob_prop(A, x0, dt, T):
    """ Simulate the propagation of probabilities given the transition matrix A,
      with initial state x0, for a duration of T at timestep dt.

      Args:
        A (ndarray): state transition matrix
        x0 (ndarray): state probabilities at time 0
        dt (scalar): timestep of the simulation
        T (scalar): total duration of the simulation

      Returns:
        ndarray, ndarray: `x` for all simulation steps and the time `t` at each step
    """

    # Initialize variables
    t = np.arange(0, T, dt)
    x = x0  # x at time t_0

    # Step through the system in time
    for k in range(len(t) - 1):
        # Compute the state of x at time k+1
        x_kp1 = np.dot(A, x[-1, :])

        # Stack (append) this new state onto x to keep track of x through time steps
        x = np.vstack((x, x_kp1))

    return x, t


def section2_distributional_perspective():
    # Set parameters
    T = 500  # total Time duration
    dt = 0.1  # timestep of our simulation

    c2o = 0.02
    o2c = 0.1

    # same parameters as above
    # c: closed rate
    # o: open rate
    c = 0.02
    o = 0.1
    A = np.array([[1 - c * dt, o * dt],
                  [c * dt, 1 - o * dt]])

    # Initial condition: start as Closed
    x0 = np.array([[1, 0]])

    # Simulate probabilities propagation
    x, t = simulate_prob_prop(A, x0, dt, T)

    # Visualize
    plot_state_probabilities(t, x)

    print(f"Probability of state c2o: {(c2o / (c2o + o2c)):.3f}")


def section3_equilibrium():
    # Set parameters
    T = 500  # total Time duration
    dt = 0.1  # timestep of our simulation

    # same parameters as above
    # c: closed rate
    # o: open rate
    c = 0.02
    o = 0.1
    A = np.array([[1 - c * dt, o * dt],
                  [c * dt, 1 - o * dt]])

    # compute the eigendecomposition of A
    lam, v = np.linalg.eig(A)

    # print the 2 eigenvalues
    print(f"Eigenvalues: {lam}")

    # print the 2 eigenvectors
    eigenvector1 = v[:, 0]
    eigenvector2 = v[:, 1]
    print(f"Eigenvector 1: {eigenvector1}")
    print(f"Eigenvector 2: {eigenvector2}")

    """
    1) Whichever eigenvalue is 1 is the stable solution. There should be another
    eigenvalue that is <1, which means it is decaying and goes away after the
    transient period.

    2) The eigenvector corresponding to this eigenvalue is the stable solution.

    3) To see this, we need to normalize this eigenvector so that its 2 elements
    sum to one, then we would see that the two numbers correspond to
    [P(open), P(closed)] at equilibrium -- hopefully these are exactly the
    equilibrium solutions observed in Section 2.
    """
    # whichever eigenvalue is 1, the other one makes no sense
    print(eigenvector1 / eigenvector1.sum())
    print(eigenvector2 / eigenvector2.sum())


if __name__ == '__main__':
    # section1_telegraph_process()
    # section2_distributional_perspective()
    section3_equilibrium()
