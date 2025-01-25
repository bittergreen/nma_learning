import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

from compneuro.w1.d2_model_fitting.utils import plot_density_image

theta = 1.2


def gen_simulated_data():
    # setting a fixed seed to our random number generator ensures we will always
    # get the same psuedorandom number sequence
    np.random.seed(121)

    # Let's set some parameters
    theta = 1.2
    n_samples = 30

    # Draw x and then calculate y
    x = 10 * np.random.rand(n_samples)  # sample from a uniform distribution over [0,10)
    noise = np.random.randn(n_samples)  # sample from a standard normal distribution
    y = theta * x + noise

    """
    # Plot the results
    fig, ax = plt.subplots()
    ax.scatter(x, y)  # produces a scatter plot
    ax.set(xlabel='x', ylabel='y')
    plt.show()
    """
    return x, y


def mse(x, y, theta_hat):
    """Compute the mean squared error

      Args:
        x (ndarray): An array of shape (samples,) that contains the input values.
        y (ndarray): An array of shape (samples,) that contains the corresponding
          measurement values to the inputs.
        theta_hat (float): An estimate of the slope parameter

      Returns:
        float: The mean squared error of the data with the estimated parameter.
    """
    # Compute the estimated y
    y_hat = theta_hat * x

    # Compute mean squared error
    mse = np.mean((y - y_hat) ** 2)

    return mse


def find_minimum_1():
    # @markdown Execute this cell to loop over theta_hats, compute MSE, and plot results

    # Loop over different thetas, compute MSE for each
    theta_hat_grid = np.linspace(-2.0, 4.0)
    errors = np.zeros(len(theta_hat_grid))
    for i, theta_hat in enumerate(theta_hat_grid):
        errors[i] = mse(x, y, theta_hat)

    # Find theta that results in lowest error
    best_error = np.min(errors)
    theta_hat = theta_hat_grid[np.argmin(errors)]

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(theta_hat_grid, errors, '-o', label='MSE', c='C1')
    ax.axvline(theta, color='g', ls='--', label=r"$\theta_{True}$")
    ax.axvline(theta_hat, color='r', ls='-', label=r"$\hat{{\theta}}_{MSE}$")
    ax.set(
        title=fr"Best fit: $\hat{{\theta}}$ = {theta_hat:.2f}, MSE = {best_error:.2f}",
        xlabel=r"$\hat{{\theta}}$",
        ylabel='MSE')
    ax.legend()
    plt.show()


def solve_normal_eqn(x, y):
    """Solve the normal equations to produce the value of theta_hat that minimizes
        MSE.

        Args:
        x (ndarray): An array of shape (samples,) that contains the input values.
        y (ndarray): An array of shape (samples,) that contains the corresponding
          measurement values to the inputs.

      Returns:
        float: the value for theta_hat arrived from minimizing MSE
    """
    # Compute theta_hat analytically
    theta_hat = (x.T @ y) / (x.T @ x)

    return theta_hat


def plot_prob(x, y):
    # @markdown Execute this cell to visualize p(y|x, theta=1.2)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    # Invokes helper function to generate density image plots from data and parameters
    im = plot_density_image(x, y, 1.2, ax=ax1)
    plt.colorbar(im, ax=ax1)
    ax1.axvline(8, color='k')
    ax1.set(title=r'p(y | x, $\theta$=1.2)')

    # Plot pdf for given x
    ylim = ax1.get_ylim()
    yy = np.linspace(ylim[0], ylim[1], 50)
    ax2.plot(yy, stats.norm(theta * 8, 1).pdf(yy), color='orange', linewidth=2)
    ax2.set(
        title=r'p(y|x=8, $\theta$=1.2)',
        xlabel='y',
        ylabel='probability density')
    plt.show()


def likelihood(theta_hat, xi, yi):
    """The likelihood function for a linear model with noise sampled from a
    Gaussian distribution with zero mean and unit variance.

      Args:
        theta_hat (float): An estimate of the slope parameter.
        x (ndarray): An array of shape (samples,) that contains the input values.
        y (ndarray): An array of shape (samples,) that contains the corresponding
          measurement values to the inputs.

      Returns:
        ndarray: the likelihood values for the theta_hat estimate
    """
    sigma = 1

    # Compute Gaussian likelihood
    pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * (yi - theta_hat * xi) ** 2)

    return pdf


def sum_log_likelihood(theta_hat, x, y):
    l = 0
    for i in range(len(x)):
        l += np.log(likelihood(theta_hat, x[i], y[i]))
    return l


def plot_likelihood():
    theta_hats = [0.5, 1.0, 2.2]
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    for theta_hat, ax in zip(theta_hats, axes):
        ll = np.sum(np.log(likelihood(theta_hat, x, y)))  # log likelihood
        im = plot_density_image(x, y, theta_hat, ax=ax)
        ax.scatter(x, y)
        ax.set(title=fr'$\hat{{\theta}}$ = {theta_hat}, log likelihood: {ll:.2f}')
    plt.colorbar(im, ax=ax)
    plt.show()


def resample_with_replacement(x, y):
    """Resample data points with replacement from the dataset of `x` inputs and
      `y` measurements.

      Args:
        x (ndarray): An array of shape (samples,) that contains the input values.
        y (ndarray): An array of shape (samples,) that contains the corresponding
          measurement values to the inputs.

      Returns:
        ndarray, ndarray: The newly resampled `x` and `y` data points.
    """
    # Get array of indices for resampled points
    sample_idx = np.random.choice(len(x), size=len(x), replace=True)

    # Sample from x and y according to sample_idx
    x_ = x[sample_idx]
    y_ = y[sample_idx]

    return x_, y_


def bootstrap_estimates(x, y, n=2000):
    """Generate a set of theta_hat estimates using the bootstrap method.

      Args:
        x (ndarray): An array of shape (samples,) that contains the input values.
        y (ndarray): An array of shape (samples,) that contains the corresponding
          measurement values to the inputs.
        n (int): The number of estimates to compute

      Returns:
        ndarray: An array of estimated parameters with size (n,)
    """
    theta_hats = np.zeros(n)

    # Loop over number of estimates
    for i in range(n):
        # Resample x and y
        x_, y_ = resample_with_replacement(x, y)

        # Compute theta_hat for this sample
        theta_hats[i] = solve_normal_eqn(x_, y_)

    return theta_hats


def visualize_all_shit():
    # @markdown Execute this cell to visualize all potential models

    fig, ax = plt.subplots()

    # For each theta_hat, plot model
    theta_hats = bootstrap_estimates(x, y, n=2000)
    for i, theta_hat in enumerate(theta_hats):
        y_hat = theta_hat * x
        ax.plot(x, y_hat, c='r', alpha=0.01, label='Resampled Fits' if i == 0 else '')

    # Plot observed data
    ax.scatter(x, y, label='Observed')

    # Plot true fit data
    y_true = theta * x
    ax.plot(x, y_true, 'g', linewidth=2, label='True Model')

    ax.set(
        title='Bootstrapped Slope Estimation',
        xlabel='x',
        ylabel='y'
    )

    # Change legend line alpha property
    handles, labels = ax.get_legend_handles_labels()
    handles[0].set_alpha(1)

    ax.legend()
    plt.show()


def bootstrap_confidence_intervals():
    # @markdown Execute this cell to plot bootstrapped CI

    theta_hats = bootstrap_estimates(x, y, n=2000)
    print(f"mean = {np.mean(theta_hats):.2f}, std = {np.std(theta_hats):.2f}")

    fig, ax = plt.subplots()
    ax.hist(theta_hats, bins=20, facecolor='C1', alpha=0.75)
    ax.axvline(theta, c='g', label=r'True $\theta$')
    ax.axvline(np.percentile(theta_hats, 50), color='r', label='Median')
    ax.axvline(np.percentile(theta_hats, 2.5), color='b', label='95% CI')
    ax.axvline(np.percentile(theta_hats, 97.5), color='b')
    ax.legend()
    ax.set(
        title='Bootstrapped Confidence Interval',
        xlabel=r'$\hat{{\theta}}$',
        ylabel='count',
        xlim=[1.0, 1.5]
    )
    plt.show()


if __name__ == '__main__':
    x, y = gen_simulated_data()
    # find_minimum_1()
    # theta_hat = solve_normal_eqn(x, y)
    # y_hat = theta_hat * x
    # plot_observed_vs_predicted(x, y, y_hat, theta_hat)
    # plot_prob(x, y)
    # print(likelihood(1.0, x[1], y[1]))
    # plot_likelihood()
    print(sum_log_likelihood(1.2, x, y))
    # x_, y_ = resample_with_replacement(x, y)

    # plot_original_and_resample(x, y, x_, y_)
    # visualize_all_shit()
    bootstrap_confidence_intervals()

