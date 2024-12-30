import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# @title Plotting Functions

def plot_observed_vs_predicted(x, y, y_hat, theta_hat):
    """ Plot observed vs predicted data

    Args:
      x (ndarray): observed x values
      y (ndarray): observed y values
      y_hat (ndarray): predicted y values
      theta_hat (ndarray):
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, label='Observed')  # our data scatter plot
    ax.plot(x, y_hat, color='r', label='Fit')  # our estimated model
    # plot residuals
    ymin = np.minimum(y, y_hat)
    ymax = np.maximum(y, y_hat)
    ax.vlines(x, ymin, ymax, 'g', alpha=0.5, label='Residuals')
    ax.set(
        title=fr"$\hat{{\theta}}$ = {theta_hat:0.2f}, MSE = {np.mean((y - y_hat) ** 2):.2f}",
        xlabel='x',
        ylabel='y'
    )
    ax.legend()
    plt.show()


# @title Plotting Functions
def plot_density_image(x, y, theta, sigma=1, ax=None):
    """ Plots probability distribution of y given x, theta, and sigma

      Args:

        x (ndarray): An array of shape (samples,) that contains the input values.
        y (ndarray): An array of shape (samples,) that contains the corresponding
          measurement values to the inputs.
        theta (float): Slope parameter
        sigma (float): standard deviation of Gaussian noise

    """

    # plot the probability density of p(y|x,theta)
    if ax is None:
        fig, ax = plt.subplots()

    xmin, xmax = np.floor(np.min(x)), np.ceil(np.max(x))
    ymin, ymax = np.floor(np.min(y)), np.ceil(np.max(y))
    xx = np.linspace(xmin, xmax, 50)
    yy = np.linspace(ymin, ymax, 50)

    surface = np.zeros((len(yy), len(xx)))
    for i, x_i in enumerate(xx):
        surface[:, i] = stats.norm(theta * x_i, sigma).pdf(yy)

    ax.set(xlabel='x', ylabel='y')

    return ax.imshow(surface, origin='lower', aspect='auto', vmin=0, vmax=None,
                     cmap=plt.get_cmap('Wistia'),
                     extent=[xmin, xmax, ymin, ymax])


# @title Plotting Functions

def plot_original_and_resample(x, y, x_, y_):
    """ Plot the original sample and the resampled points from this sample.

      Args:
        x (ndarray): An array of shape (samples,) that contains the input values.
        y (ndarray): An array of shape (samples,) that contains the corresponding
          measurement values to the inputs.
        x_ (ndarray): An array of shape (samples,) with a subset of input values from x
        y_ (ndarray): An array of shape (samples,) with a the corresponding subset
          of measurement values as x_ from y

    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.scatter(x, y)
    ax1.set(title='Original', xlabel='x', ylabel='y')

    ax2.scatter(x_, y_, color='c')

    ax2.set(title='Resampled', xlabel='x', ylabel='y',
            xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
    plt.show()


# @title Plotting Functions

def evaluate_fits(order_list, mse_list):
    """ Compare the quality of multiple polynomial fits
  by plotting their MSE values.

  Args:
    order_list (list): list of the order of polynomials to be compared
    mse_list (list): list of the MSE values for the corresponding polynomial fit
  """
    fig, ax = plt.subplots()
    ax.bar(order_list, mse_list)
    ax.set(title='Comparing Polynomial Fits', xlabel='Polynomial order', ylabel='MSE')
    plt.show()


# @title Plotting Functions

def plot_MSE_poly_fits(mse_train, mse_test, max_order):
    """
    Plot the MSE values for various orders of polynomial fits on the same bar
    graph

        Args:
          mse_train (ndarray): an array of MSE values for each order of polynomial fit
          over the training data
          mse_test (ndarray): an array of MSE values for each order of polynomial fit
          over the test data
          max_order (scalar): max order of polynomial fit
    """
    fig, ax = plt.subplots()
    width = .35

    ax.bar(np.arange(max_order + 1) - width / 2,
           mse_train, width, label="train MSE")
    ax.bar(np.arange(max_order + 1) + width / 2,
           mse_test, width, label="test MSE")

    ax.legend()
    ax.set(xlabel='Polynomial order', ylabel='MSE',
           title='Comparing polynomial fits')
    plt.show()


# @title Plotting Functions

def plot_cross_validate_MSE(mse_all, max_order, n_splits):
    """ Plot the MSE values for the K_fold cross validation

  Args:
    mse_all (ndarray): an array of size (number of splits, max_order + 1)
  """
    plt.figure()
    plt.boxplot(mse_all, labels=np.arange(0, max_order + 1))

    plt.xlabel('Polynomial Order')
    plt.ylabel('Validation MSE')
    plt.title(f'Validation MSE over {n_splits} splits of the data')
    plt.show()


def plot_AIC(order_list, AIC_list):
    """ Plot the AIC value for fitted polynomials of various orders

  Args:
    order_list (list): list of fitted polynomial orders
    AIC_list (list): list of AIC values corresponding to each polynomial model on order_list
  """
    plt.bar(order_list, AIC_list)
    plt.ylabel('AIC')
    plt.xlabel('polynomial order')
    plt.title('comparing polynomial fits')
    plt.show()
