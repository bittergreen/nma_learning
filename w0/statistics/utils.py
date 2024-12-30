import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# @title Plotting Functions
def plot_random_sample(dataX, dataY, figtitle=None):
    """ Plot the random sample between 0 and 1 for both the x and y axes.

    Args:
      x (ndarray): array of x coordinate values across the random sample
      y (ndarray): array of y coordinate values across the random sample
      figtitle (str): title of histogram plot (default is no title)

    Returns:
      Nothing.
  """
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xlim([-0.25, 1.25])  # set x and y axis range to be a bit less than 0 and greater than 1
    plt.ylim([-0.25, 1.25])
    plt.scatter(dataX, dataY)
    if figtitle is not None:
        fig.suptitle(figtitle, size=16)
    plt.show()


def plot_random_walk(x, y, figtitle=None):
    """ Plots the random walk within the range 0 to 1 for both the x and y axes.

    Args:
      x (ndarray): array of steps in x direction
      y (ndarray): array of steps in y direction
      figtitle (str): title of histogram plot (default is no title)

    Returns:
      Nothing.
  """
    fig, ax = plt.subplots()
    plt.plot(x, y, 'b-o', alpha=0.5)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    ax.set_xlabel('x location')
    ax.set_ylabel('y location')
    plt.plot(x[0], y[0], 'go')
    plt.plot(x[-1], y[-1], 'ro')

    if figtitle is not None:
        fig.suptitle(figtitle, size=16)
    plt.show()


def my_plot_single(x, px):
    """
  Plots normalized Gaussian distribution

    Args:
        x (numpy array of floats):     points at which the likelihood has been evaluated
        px (numpy array of floats):    normalized probabilities for prior evaluated at each `x`

    Returns:
        Nothing.
  """
    if px is None:
        px = np.zeros_like(x)

    fig, ax = plt.subplots()
    ax.plot(x, px, '-', color='C2', linewidth=2, label='Prior')
    ax.legend()
    ax.set_ylabel('Probability')
    ax.set_xlabel('Orientation (Degrees)')
    plt.show()


# @title Plotting functions

def plot_hist(data, xlabel, figtitle=None, num_bins=None):
    """ Plot the given data as a histogram.

    Args:
      data (ndarray): array with data to plot as histogram
      xlabel (str): label of x-axis
      figtitle (str): title of histogram plot (default is no title)
      num_bins (int): number of bins for histogram (default is 10)

    Returns:
      count (ndarray): number of samples in each histogram bin
      bins (ndarray): center of each histogram bin
  """
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    if num_bins is not None:
        count, bins, _ = plt.hist(data, max(data), bins=num_bins)
    else:
        count, bins, _ = plt.hist(data, max(data))  # 10 bins default
    if figtitle is not None:
        fig.suptitle(figtitle, size=16)
    plt.show()
    return count, bins


def plot_gaussian_samples_true(samples, xspace, mu, sigma, xlabel, ylabel):
    """ Plot a histogram of the data samples on the same plot as the gaussian
  distribution specified by the give mu and sigma values.

    Args:
      samples (ndarray): data samples for gaussian distribution
      xspace (ndarray): x values to sample from normal distribution
      mu (scalar): mean parameter of normal distribution
      sigma (scalar): variance parameter of normal distribution
      xlabel (str): the label of the x-axis of the histogram
      ylabel (str): the label of the y-axis of the histogram

    Returns:
      Nothing.
  """
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # num_samples = samples.shape[0]

    count, bins, _ = plt.hist(samples, density=True)  # probability density function

    plt.plot(xspace, norm.pdf(xspace, mu, sigma), 'r-')
    plt.show()


def plot_likelihoods(likelihoods, mean_vals, variance_vals):
    """ Plot the likelihood values on a heatmap plot where the x and y axes match
  the mean and variance parameter values the likelihoods were computed for.

    Args:
      likelihoods (ndarray): array of computed likelihood values
      mean_vals (ndarray): array of mean parameter values for which the
                            likelihood was computed
      variance_vals (ndarray): array of variance parameter values for which the
                            likelihood was computed

    Returns:
      Nothing.
  """
    fig, ax = plt.subplots()
    im = ax.imshow(likelihoods)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('log likelihood', rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(mean_vals)))
    ax.set_yticks(np.arange(len(variance_vals)))
    ax.set_xticklabels(mean_vals)
    ax.set_yticklabels(variance_vals)
    ax.set_xlabel('Mean')
    ax.set_ylabel('Variance')
    plt.show()


def posterior_plot(x, likelihood=None, prior=None,
                   posterior_pointwise=None, ax=None):
    """
  Plots normalized Gaussian distributions and posterior.

    Args:
        x (numpy array of floats):         points at which the likelihood has been evaluated
        auditory (numpy array of floats):  normalized probabilities for auditory likelihood evaluated at each `x`
        visual (numpy array of floats):    normalized probabilities for visual likelihood evaluated at each `x`
        posterior (numpy array of floats): normalized probabilities for the posterior evaluated at each `x`
        ax: Axis in which to plot. If None, create new axis.

    Returns:
        Nothing.
  """
    if likelihood is None:
        likelihood = np.zeros_like(x)

    if prior is None:
        prior = np.zeros_like(x)

    if posterior_pointwise is None:
        posterior_pointwise = np.zeros_like(x)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, likelihood, '-C1', linewidth=2, label='Auditory')
    ax.plot(x, prior, '-C0', linewidth=2, label='Visual')
    ax.plot(x, posterior_pointwise, '-C2', linewidth=2, label='Posterior')
    ax.legend()
    ax.set_ylabel('Probability')
    ax.set_xlabel('Orientation (Degrees)')
    plt.show()

    return ax


def plot_classical_vs_bayesian_normal(num_points, mu_classic, var_classic,
                                      mu_bayes, var_bayes):
    """ Helper function to plot optimal normal distribution parameters for varying
  observed sample sizes using both classic and Bayesian inference methods.

    Args:
      num_points (int): max observed sample size to perform inference with
      mu_classic (ndarray): estimated mean parameter for each observed sample size
                                using classic inference method
      var_classic (ndarray): estimated variance parameter for each observed sample size
                                using classic inference method
      mu_bayes (ndarray): estimated mean parameter for each observed sample size
                                using Bayesian inference method
      var_bayes (ndarray): estimated variance parameter for each observed sample size
                                using Bayesian inference method

    Returns:
      Nothing.
  """
    xspace = np.linspace(0, num_points, num_points)
    fig, ax = plt.subplots()
    ax.set_xlabel('n data points')
    ax.set_ylabel('mu')
    plt.plot(xspace, mu_classic, 'r-', label="Classical")
    plt.plot(xspace, mu_bayes, 'b-', label="Bayes")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('n data points')
    ax.set_ylabel('sigma^2')
    plt.plot(xspace, var_classic, 'r-', label="Classical")
    plt.plot(xspace, var_bayes, 'b-', label="Bayes")
    plt.legend()
    plt.show()


