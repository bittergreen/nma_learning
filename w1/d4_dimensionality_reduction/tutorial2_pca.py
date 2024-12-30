# Imports
import numpy as np
import matplotlib.pyplot as plt

from w1.d4_dimensionality_reduction.utils import calculate_cov_matrix, get_data, plot_basis_vectors, \
    sort_evals_descending, plot_data_new_basis, plot_eigenvalues


def get_sample_cov_matrix(X):
    """
      Returns the sample covariance matrix of data X

      Args:
        X (numpy array of floats) : Data matrix each column corresponds to a
                                    different random variable

      Returns:
        (numpy array of floats)   : Covariance matrix
    """
    # Subtract the mean of X
    X = X - np.mean(X, 0).reshape(1, 2)

    # Calculate the covariance matrix (hint: use np.matmul)
    cov_matrix = (1 / X.shape[0]) * X.T @ X

    return cov_matrix


def compute_basis_vectors(cov_matrix):
    # Calculate the eigenvalues and eigenvectors
    evals, evectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvalues in descending order
    evals, evectors = sort_evals_descending(evals, evectors)

    # Visualize
    # plot_basis_vectors(X, evectors)
    return evals, evectors


def pca(X):
    """
      Sorts eigenvalues and eigenvectors in decreasing order.

      Args:
        X (numpy array of floats): Data matrix each column corresponds to a
                                   different random variable

      Returns:
        (numpy array of floats)  : Data projected onto the new basis
        (numpy array of floats)  : Vector of eigenvalues
        (numpy array of floats)  : Corresponding matrix of eigenvectors

    """
    # Calculate the sample covariance matrix
    cov_matrix = get_sample_cov_matrix(X)

    # Calculate the eigenvalues and eigenvectors
    evals, evectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvalues in descending order
    evals, evectors = sort_evals_descending(evals, evectors)

    # Project the data onto the new eigenvector basis
    score = X @ evectors

    return score, evectors, evals


if __name__ == '__main__':
    # Set parameters
    np.random.seed(2020)  # set random seed
    variance_1 = 1
    variance_2 = 1
    corr_coef = 1

    # Calculate covariance matrix
    cov_matrix = calculate_cov_matrix(variance_1, variance_2, corr_coef)
    print(cov_matrix)

    # Generate data with that covariance matrix
    X = get_data(cov_matrix)

    # Get sample covariance matrix
    sample_cov_matrix = get_sample_cov_matrix(X)
    print(sample_cov_matrix)

    compute_basis_vectors(sample_cov_matrix)

    # Perform PCA on the data matrix X
    score, evectors, evals = pca(X)

    # Plot the data projected into the new basis
    plot_data_new_basis(score)
    plot_eigenvalues(evals)
