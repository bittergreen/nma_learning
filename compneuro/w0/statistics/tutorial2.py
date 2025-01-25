import numpy as np
import scipy as sp
from numpy.random import default_rng

from utils import *


def markov_chain():
    # Transition matrix
    transition_matrix = np.array([[0.2, 0.6, 0.2], [.6, 0.3, 0.1], [0.8, 0.2, 0]])

    # Initial state, p0
    p0 = np.array([0, 1, 0])

    # Compute the probabilities 4 transitions later (use np.linalg.matrix_power to raise a matrix a power)
    p4 = p0 @ np.linalg.matrix_power(transition_matrix, 4)

    # The second area is indexed as 1 (Python starts indexing at 0)
    print(f"The probability the rat will be in area 2 after 4 transitions is: {p4[1]}")

    # Initialize random initial distribution
    p_random = np.ones((1, 3)) / 3

    # Fill in the missing line to get the state matrix after 100 transitions, like above
    p_average_time_spent = p0 @ np.linalg.matrix_power(transition_matrix, 100)
    print(f"The proportion of time spend by the rat in each of the three states is: {p_average_time_spent}")


def compute_likelihood_normal(x, mean_val, standard_dev_val):
    """ Computes the log-likelihood values given a observed data sample x, and
    potential mean and variance values for a normal distribution

    Args:
      x (ndarray): 1-D array with all the observed data
      mean_val (scalar): value of mean for which to compute likelihood
      standard_dev_val (scalar): value of variance for which to compute likelihood

    Returns:
      likelihood (scalar): value of likelihood for this combination of means/variances
    """
    # Get probability of each data point (use norm.pdf from scipy stats)
    p_data = norm.pdf(x, mean_val, standard_dev_val)

    # Compute likelihood (sum over the log of the probabilities)
    likelihood = np.sum(np.log(p_data))

    return likelihood


def compute_likelihood():
    # Set random seed
    np.random.seed(0)

    # Generate data
    true_mean = 5
    true_standard_dev = 1
    n_samples = 1000
    x = np.random.normal(true_mean, true_standard_dev, size=(n_samples,))

    # Compute likelihood for a guessed mean/standard dev
    guess_mean = 5
    guess_standard_dev = 1
    likelihood = compute_likelihood_normal(x, guess_mean, guess_standard_dev)
    print(likelihood)


def plot_likelihood():
    np.random.seed(0)

    # Generate data
    true_mean = 5
    true_standard_dev = 1
    n_samples = 1000
    x = np.random.normal(true_mean, true_standard_dev, size=(n_samples,))

    # Compute likelihood for different mean/variance values
    mean_vals = np.linspace(1, 10, 10)  # potential mean values to ry
    standard_dev_vals = np.array([0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 5])  # potential variance values to try

    # Initialise likelihood collection array
    likelihood = np.zeros((mean_vals.shape[0], standard_dev_vals.shape[0]))

    # Compute the likelihood for observing the gvien data x assuming
    # each combination of mean and variance values
    for idxMean in range(mean_vals.shape[0]):
        for idxVar in range(standard_dev_vals.shape[0]):
            likelihood[idxVar, idxMean] = sum(np.log(norm.pdf(x, mean_vals[idxMean],
                                                              standard_dev_vals[idxVar])))

    # Uncomment once you've generated the samples and compute likelihoods
    xspace = np.linspace(0, 10, 100)
    plot_likelihoods(likelihood, mean_vals, standard_dev_vals)


def negLogLike(theta, x):
    """ Function for computing the negative log-likelihood given the observed data
      and given parameter values stored in theta.

      Args:
        theta (ndarray): normal distribution parameters
                        (mean is theta[0], standard deviation is theta[1])
        x (ndarray): array with observed data points

      Returns:
        Calculated negative Log Likelihood value!
    """
    return -sum(np.log(norm.pdf(x, theta[0], theta[1])))


def optimize():
    # Set random seed
    np.random.seed(0)

    # Generate data
    true_mean = 5
    true_standard_dev = 1
    n_samples = 1000
    x = np.random.normal(true_mean, true_standard_dev, size=(n_samples,))

    # Define bounds, var has to be positive
    bnds = ((None, None), (0, None))

    # Optimize with scipy!
    optimal_parameters = sp.optimize.minimize(negLogLike, (2, 2), args=x, bounds=bnds)
    print(f"The optimal mean estimate is: {optimal_parameters.x[0]}")
    print(f"The optimal standard deviation estimate is: {optimal_parameters.x[1]}")


# @markdown Execute to visualize inference

def classic_vs_bayesian_normal(mu, sigma, num_points, prior):
    """ Compute both classical and Bayesian inference processes over the range of
  data sample sizes (num_points) for a normal distribution with parameters
  mu,sigma for comparison.

  Args:
    mu (scalar): the mean parameter of the normal distribution
    sigma (scalar): the standard deviation parameter of the normal distribution
    num_points (int): max number of points to use for inference
    prior (ndarray): prior data points for Bayesian inference

  Returns:
    mean_classic (ndarray): estimate mean parameter via classic inference
    var_classic (ndarray): estimate variance parameter via classic inference
    mean_bayes (ndarray): estimate mean parameter via Bayesian inference
    var_bayes (ndarray): estimate variance parameter via Bayesian inference
  """

    # Initialize the classical and Bayesian inference arrays that will estimate
    # the normal parameters given a certain number of randomly sampled data points
    mean_classic = np.zeros(num_points)
    var_classic = np.zeros(num_points)

    mean_bayes = np.zeros(num_points)
    var_bayes = np.zeros(num_points)

    for nData in range(num_points):
        random_num_generator = default_rng(0)
        x = random_num_generator.normal(mu, sigma, nData + 1)

        # Compute the mean of those points and set the corresponding array entry to this value
        mean_classic[nData] = np.mean(x)

        # Compute the variance of those points and set the corresponding array entry to this value
        var_classic[nData] = np.var(x)

        # Bayesian inference with the given prior is performed below for you
        xsupp = np.hstack((x, prior))
        mean_bayes[nData] = np.mean(xsupp)
        var_bayes[nData] = np.var(xsupp)

    return mean_classic, var_classic, mean_bayes, var_bayes


def compare():
    # Set random seed
    np.random.seed(0)

    # Set normal distribution parameters, mu and sigma
    mu = 5
    sigma = 1

    # Set the prior to be two new data points, 4 and 6, and print the mean and variance
    prior = np.array((4, 6))
    print("The mean of the data comprising the prior is: " + str(np.mean(prior)))
    print("The variance of the data comprising the prior is: " + str(np.var(prior)))

    mean_classic, var_classic, mean_bayes, var_bayes = classic_vs_bayesian_normal(mu, sigma, 60, prior)
    plot_classical_vs_bayesian_normal(60, mean_classic, var_classic, mean_bayes, var_bayes)


def plotFnc(p, n, priorL, priorR):
    # Set random seed
    np.random.seed(1)
    # sample from binomial
    numL = np.random.binomial(n, p, 1)
    numR = n - numL
    stepSize = 0.001
    x = np.arange(0, 1, stepSize)
    betaPdf = sp.stats.beta.pdf(x, numL + priorL, numR + priorR)
    betaPrior = sp.stats.beta.pdf(x, priorL, priorR)
    print("number of left " + str(numL))
    print("number of right " + str(numR))
    print(" ")
    print("max likelihood " + str(numL / (numL + numR)))
    print(" ")
    print("max posterior " + str(x[np.argmax(betaPdf)]))
    print("mean posterior " + str(np.mean(betaPdf * x)))

    print(" ")

    with plt.xkcd():
        # rng.beta()
        fig, ax = plt.subplots()
        plt.rcParams.update({'font.size': 22})
        ax.set_xlabel('p')
        ax.set_ylabel('probability density')
        plt.plot(x, betaPdf, label="Posterior")
        plt.plot(x, betaPrior, label="Prior")
        # print(int(len(betaPdf)/2))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # markov_chain()
    # compute_likelihood()
    # plot_likelihood()
    # optimize()
    # compare()
    plotFnc(p=0.2, n=30, priorL=5, priorR=5)
