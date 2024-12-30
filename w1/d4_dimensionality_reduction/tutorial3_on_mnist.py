# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from w1.d4_dimensionality_reduction.utils import plot_MNIST_sample, pca, plot_eigenvalues, plot_variance_explained, \
    plot_MNIST_reconstruction, plot_MNIST_weights


def get_variance_explained(evals):
    """
      Calculates variance explained from the eigenvalues.

      Args:
        evals (numpy array of floats) : Vector of eigenvalues

      Returns:
        (numpy array of floats)       : Vector of variance explained

    """
    # Cumulatively sum the eigenvalues
    csum = np.cumsum(evals)

    # Normalize by the sum of eigenvalues
    variance_explained = csum / np.sum(evals)

    return variance_explained


def reconstruct_data(score, evectors, X_mean, K):
    """
      Reconstruct the data based on the top K components.

      Args:
        score (numpy array of floats)    : Score matrix
        evectors (numpy array of floats) : Matrix of eigenvectors
        X_mean (numpy array of floats)   : Vector corresponding to data mean
        K (scalar)                       : Number of components to include

      Returns:
        (numpy array of floats)          : Matrix of reconstructed data

    """
    # Reconstruct the data from the score and eigenvectors
    # Don't forget to add the mean!!
    X_reconstructed = score[:, :K] @ evectors[:, :K].T + X_mean

    return X_reconstructed


if __name__ == '__main__':
    # GET mnist data
    mnist = fetch_openml(name='mnist_784', as_frame=False, parser='auto')
    X = mnist.data

    # Visualize
    # plot_MNIST_sample(X)

    # Perform PCA
    score, evectors, evals = pca(X)

    # Plot the eigenvalues
    # plot_eigenvalues(evals, xlimit=True)  # limit x-axis up to 100 for zooming

    # Calculate the variance explained
    variance_explained = get_variance_explained(evals)

    # Visualize
    # plot_variance_explained(variance_explained)

    K = 100  # data dimensions

    # Reconstruct the data based on all components
    X_mean = np.mean(X, 0)
    X_reconstructed = reconstruct_data(score, evectors, X_mean, K)

    # Plot the data and reconstruction
    # plot_MNIST_reconstruction(X, X_reconstructed, K)

    # Plot the weights of the first principal component
    plot_MNIST_weights(evectors[:, 0])

