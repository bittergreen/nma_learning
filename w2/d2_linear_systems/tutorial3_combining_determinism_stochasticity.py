# Imports
import numpy as np
import matplotlib.pyplot as plt

from w2.d2_linear_systems.utils import plot_random_walk_sims, plot_mean_var_by_timestep, plot_ddm, var_comparison_plot


def random_walk_simulator(N, T, mu=0, sigma=1):
    """Simulate N random walks for T time points. At each time point, the step
      is drawn from a Gaussian distribution with mean mu and standard deviation
      sigma.

      Args:
        T (integer) : Duration of simulation in time steps
        N (integer) : Number of random walks
        mu (float) : mean of step distribution
        sigma (float) : standard deviation of step distribution

      Returns:
        (numpy array) : NxT array in which each row corresponds to trajectory
    """
    # generate all the random steps for all steps in all simulations in one go
    # produces a N x T array
    steps = np.random.normal(mu, sigma, size=(N, T))

    # compute the cumulative sum of all the steps over the time axis
    sim = np.cumsum(steps, axis=1)

    return sim


def diffusive_process(sim):
    # @markdown Execute to visualize distribution of bacteria positions
    fig = plt.figure()
    # look at the distribution of positions at different times
    for i, t in enumerate([1000, 2500, 10000]):
        # get mean and standard deviation of distribution at time t
        mu = sim[:, t - 1].mean()
        sig2 = sim[:, t - 1].std()

        # make a plot label
        mytitle = '$t=${time:d} ($\mu=${mu:.2f}, $\sigma=${var:.2f})'

        # plot histogram
        plt.hist(sim[:, t - 1],
                 color=['blue', 'orange', 'black'][i],
                 # make sure the histograms have the same bins!
                 bins=np.arange(-300, 300, 20),
                 # make histograms a little see-through
                 alpha=0.6,
                 # draw second histogram behind the first one
                 zorder=3 - i,
                 label=mytitle.format(time=t, mu=mu, var=sig2))

        plt.xlabel('position x')

        # plot range
        plt.xlim([-500, 250])

        # add legend
        plt.legend(loc=2)

    # add title
    plt.title(r'Distribution of trajectory positions at time $t$')
    plt.show()


def section1_random_walks():
    np.random.seed(2020)  # set random seed

    # simulate 1000 random walks for 10000 time steps
    sim = random_walk_simulator(1000, 10000, mu=0, sigma=1)

    # take a peek at the first 10 simulations
    plot_random_walk_sims(sim, nsims=10)

    diffusive_process(sim)

    sim = random_walk_simulator(5000, 1000, mu=0, sigma=1)

    # Compute mean
    mu = np.mean(sim, axis=0)

    # Compute variance
    var = np.var(sim, axis=0)

    # Visualize
    plot_mean_var_by_timestep(mu, var)


def simulate_ddm(lam, sig, x0, xinfty, T):
    """
      Simulate the drift-diffusion model with given parameters and initial condition.
      Args:
        lam (scalar): decay rate
        sig (scalar): standard deviation of normal distribution
        x0 (scalar): initial condition (x at time 0)
        xinfty (scalar): drift towards convergence in the limit
        T (scalar): total duration of the simulation (in steps)

      Returns:
        ndarray, ndarray: `x` for all simulation steps and the time `t` at each step
  """

    # initialize variables
    t = np.arange(0, T, 1.)
    x = np.zeros_like(t)
    x[0] = x0

    # Step through in time
    for k in range(len(t) - 1):
        # update x at time k+1 with a deterministic and a stochastic component
        # hint: the deterministic component will be like above, and
        #   the stochastic component is drawn from a scaled normal distribution
        x[k + 1] = xinfty + lam * (x[k] - xinfty) + sig * np.random.normal(0, 1)

    return t, x


def section2_ornstein_uhlenbeck_process():
    lam = 0.9  # decay rate
    sig = 0.1  # standard deviation of diffusive process
    T = 500  # total Time duration in steps
    x0 = 4.  # initial condition of x at time 0
    xinfty = 1.  # x drifts towards this value in long time

    # Plot x as it evolves in time
    np.random.seed(2020)
    t, x = simulate_ddm(lam, sig, x0, xinfty, T)
    plot_ddm(t, x, xinfty, lam, x0)


def ddm(T, x0, xinfty, lam, sig):
    t = np.arange(0, T, 1.)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t)-1):
        x[k+1] = xinfty + lam * (x[k] - xinfty) + sig * np.random.normal(0, 1)

    return t, x


# computes equilibrium variance of ddm
# returns variance
def ddm_eq_var(T, x0, xinfty, lam, sig):
    t, x = ddm(T, x0, xinfty, lam, sig)

    # returns variance of the second half of the simulation
    # this is a hack: assumes system has settled by second half
    return x[-round(T/2):].var()


def section3_variance_of_ou():
    np.random.seed(2020)  # set random seed

    x0 = 4.  # initial condition of x at time 0
    xinfty = 1.  # x drifts towards this value in long time

    # sweep through values for lambda
    lambdas = np.arange(0.05, 0.95, 0.01)
    empirical_variances = np.zeros_like(lambdas)
    analytical_variances = np.zeros_like(lambdas)

    sig = 0.87

    # compute empirical equilibrium variance
    for i, lam in enumerate(lambdas):
        empirical_variances[i] = ddm_eq_var(5000, x0, xinfty, lambdas[i], sig)

    # Hint: you can also do this in one line outside the loop!
    analytical_variances = sig**2 / (1 - lambdas**2)

    # Plot the empirical variance vs analytical variance
    var_comparison_plot(empirical_variances, analytical_variances)


if __name__ == '__main__':
    # section1_random_walks()
    # section2_ornstein_uhlenbeck_process()
    section3_variance_of_ou()
