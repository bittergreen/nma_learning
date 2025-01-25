# @markdown Execute this cell to get helper function `get_data`
import numpy as np

from compneuro.w1.d4_dimensionality_reduction.utils import plot_basis_vectors, plot_data_new_basis


def get_data(cov_matrix):
    """
      Returns a matrix of 1000 samples from a bivariate, zero-mean Gaussian.

      Note that samples are sorted in ascending order for the first random variable

      Args:
        cov_matrix (numpy array of floats): desired covariance matrix

      Returns:
        (numpy array of floats) : samples from the bivariate Gaussian, with each
                                  column corresponding to a different random
                                  variable
    """

    mean = np.array([0, 0])
    X = np.random.multivariate_normal(mean, cov_matrix, size=1000)
    indices_for_sorting = np.argsort(X[:, 0])
    X = X[indices_for_sorting, :]

    return X


def calculate_cov_matrix(var_1, var_2, corr_coef):
    """
      Calculates the covariance matrix based on the variances and correlation
      coefficient.

      Args:
        var_1 (scalar)          : variance of the first random variable
        var_2 (scalar)          : variance of the second random variable
        corr_coef (scalar)      : correlation coefficient

      Returns:
        (numpy array of floats) : covariance matrix
    """
    # Calculate the covariance from the variances and correlation
    cov = corr_coef * np.sqrt(var_1 * var_2)

    cov_matrix = np.array([[var_1, cov], [cov, var_2]])

    return cov_matrix


def define_orthonormal_basis(u):
    """
    Calculates an orthonormal basis given an arbitrary vector u.

    Args:
    u (numpy array of floats) : arbitrary 2-dimensional vector used for new
                                basis

    Returns:
    (numpy array of floats)   : new orthonormal basis
                                columns correspond to basis vectors
    """
    # Normalize vector u
    u = u / np.sqrt(np.sum(u ** 2))

    # Calculate vector w that is orthogonal to u
    w = [-u[1], u[0]]

    # Put in matrix form
    W = np.column_stack([u, w])

    return W


def change_of_basis(X, W):
    """
      Projects data onto new basis W.

      Args:
        X (numpy array of floats) : Data matrix each column corresponding to a
                                    different random variable
        W (numpy array of floats) : new orthonormal basis columns correspond to
                                    basis vectors

      Returns:
        (numpy array of floats)    : Data matrix expressed in new basis
    """
    # Project data onto new basis described by W
    Y = X @ W

    return Y


def refresh(theta=0):
    # Set up parameters
    np.random.seed(2020)  # set random seed
    variance_1 = 1
    variance_2 = 1
    corr_coef = 0.8
    # Compute covariance matrix
    cov_matrix = calculate_cov_matrix(variance_1, variance_2, corr_coef)

    # Generate data
    X = get_data(cov_matrix)

    u = np.array([1, np.tan(theta * np.pi / 180)])
    W = define_orthonormal_basis(u)
    Y = change_of_basis(X, W)
    plot_basis_vectors(X, W)
    plot_data_new_basis(Y)


def do():
    # Set up parameters
    np.random.seed(2020)  # set random seed
    variance_1 = 1
    variance_2 = 1
    corr_coef = 0.8
    u = np.array([3, 1])

    # Compute covariance matrix
    cov_matrix = calculate_cov_matrix(variance_1, variance_2, corr_coef)

    # Generate data
    X = get_data(cov_matrix)

    # Get orthonomal basis
    W = define_orthonormal_basis(u)

    # Project data to new basis
    Y = change_of_basis(X, W)

    # Visualize
    plot_data_new_basis(Y)


if __name__ == '__main__':
    refresh(90)
