# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat

from utils import download_data, plot_stim_and_spikes, plot_glm_matrices, plot_spike_filter, plot_spikes_with_prediction


def load_data():
    download_data()
    data = loadmat('RGCdata.mat')  # loadmat is a function in scipy.io
    dt_stim = data['dtStim'].item()  # .item extracts a scalar value

    # Extract the stimulus intensity
    stim = data['Stim'].squeeze()  # .squeeze removes dimensions with 1 element

    # Extract the spike counts for one cell
    cellnum = 2
    spikes = data['SpCounts'][:, cellnum]

    # Don't use all of the timepoints in the dataset, for speed
    keep_timepoints = 20000
    stim = stim[:keep_timepoints]
    spikes = spikes[:keep_timepoints]
    # plot_stim_and_spikes(stim, spikes, dt_stim)
    return stim, spikes, dt_stim


def make_design_matrix(stim, d=25):
    """Create time-lag design matrix from stimulus intensity vector.

  Args:
    stim (1D array): Stimulus intensity at each time point.
    d (number): Number of time lags to use.

  Returns
    X (2D array): GLM design matrix with shape T, d

  """

    # Create version of stimulus vector with zeros before onset

    padded_stim = np.concatenate([np.zeros(d - 1), stim])

    # Construct a matrix where each row has the d frames of
    # the stimulus preceding and including timepoint t
    T = len(stim)  # Total number of timepoints (hint: number of stimulus frames)
    X = np.zeros((T, d))
    for t in range(T):
        X[t] = padded_stim[t:t + d]

    return X


def column_pad(stim, spikes, dt_stim):
    # Build the full design matrix
    y = spikes
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_design_matrix(stim)])

    # Get the MLE weights for the LG model
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    theta_lg = theta[1:]
    # plot_spike_filter(theta_lg, dt_stim)
    return theta_lg


def predict_spike_counts_lg(stim, spikes, d=25):
    """Compute a vector of predicted spike counts given the stimulus.

  Args:
    stim (1D array): Stimulus values at each timepoint
    spikes (1D array): Spike counts measured at each timepoint
    d (number): Number of time lags to use.

  Returns:
    yhat (1D array): Predicted spikes at each timepoint.

  """
    # Create the design matrix
    y = spikes
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_design_matrix(stim, d)])

    # Get the MLE weights for the LG model
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    # Compute predicted spike counts
    yhat = X @ theta

    return yhat


def neg_log_lik_lnp(theta, X, y):
    """Return -loglike for the Poisson GLM model.

      Args:
        theta (1D array): Parameter vector.
        X (2D array): Full design matrix.
        y (1D array): Data values.

      Returns:
        number: Negative log likelihood.

    """
    # Compute the Poisson log likelihood
    rate = np.exp(X @ theta)
    log_lik = y @ np.log(rate) - np.sum(rate)

    return -log_lik


def fit_lnp(stim, spikes, d=25):
    """Obtain MLE parameters for the Poisson GLM.

      Args:
        stim (1D array): Stimulus values at each timepoint
        spikes (1D array): Spike counts measured at each timepoint
        d (number): Number of time lags to use.

      Returns:
        1D array: MLE parameters

    """
    # Build the design matrix
    y = spikes
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_design_matrix(stim)])

    # Use a random vector of weights to start (mean 0, sd .2)
    x0 = np.random.normal(0, .2, d + 1)

    # Find parameters that minmize the negative log likelihood function
    res = minimize(neg_log_lik_lnp, x0, args=(X, y))

    return res["x"]


def fit_poisson():
    # Fit LNP model
    theta_lnp = fit_lnp(stim, spikes)

    # Visualize
    plot_spike_filter(theta_lg[1:], dt_stim, show=False, color=".5", label="LG")
    plot_spike_filter(theta_lnp[1:], dt_stim, show=False, label="LNP")
    plt.legend(loc="upper left")
    plt.show()


def predict_spike_counts_lnp(stim, spikes, theta=None, d=25):
    """Compute a vector of predicted spike counts given the stimulus.

      Args:
        stim (1D array): Stimulus values at each timepoint
        spikes (1D array): Spike counts measured at each timepoint
        theta (1D array): Filter weights; estimated if not provided.
        d (number): Number of time lags to use.

      Returns:
        yhat (1D array): Predicted spikes at each timepoint.

    """
    y = spikes
    constant = np.ones_like(spikes)
    X = np.column_stack([constant, make_design_matrix(stim)])
    if theta is None:  # Allow pre-cached weights, as fitting is slow
        theta = fit_lnp(X, y, d)

    yhat = np.exp(X @ theta)
    return yhat


if __name__ == '__main__':
    stim, spikes, dt_stim = load_data()

    # Make design matrix
    # X = make_design_matrix(stim)

    # Visualize
    # plot_glm_matrices(X, spikes, nt=50)
    theta_lg = column_pad(stim, spikes, dt_stim)

    # Predict spike counts
    # predicted_counts = predict_spike_counts_lg(stim, spikes)

    # Visualize
    # plot_spikes_with_prediction(spikes, predicted_counts, dt_stim)

    # fit_poisson()

    # Predict spike counts
    theta_lnp = fit_lnp(stim, spikes)
    yhat = predict_spike_counts_lnp(stim, spikes, theta_lnp)

    # Visualize
    plot_spikes_with_prediction(spikes, yhat, dt_stim)
