# Imports
import numpy as np
import matplotlib.pyplot as plt


def plot_data(X):
    """
  Plots bivariate data. Includes a plot of each random variable, and a scatter
  plot of their joint activity. The title indicates the sample correlation
  calculated from the data.

  Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable

  Returns:
    Nothing.
  """

    fig = plt.figure(figsize=[8, 4])
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(X[:, 0], color='k')
    plt.ylabel('Neuron 1')
    plt.title(f'Sample var 1: {np.var(X[:, 0]):.1f}')
    ax1.set_xticklabels([])
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(X[:, 1], color='k')
    plt.xlabel('Sample Number')
    plt.ylabel('Neuron 2')
    plt.title(f'Sample var 2: {np.var(X[:, 1]):.1f}')
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.plot(X[:, 0], X[:, 1], '.', markerfacecolor=[.5, .5, .5],
             markeredgewidth=0)
    ax3.axis('equal')
    plt.xlabel('Neuron 1 activity')
    plt.ylabel('Neuron 2 activity')
    plt.title(f'Sample corr: {np.corrcoef(X[:, 0], X[:, 1])[0, 1]:.1f}')
    plt.show()


def plot_basis_vectors(X, W):
    """
  Plots bivariate data as well as new basis vectors.

  Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable
    W (numpy array of floats) : Square matrix representing new orthonormal
                                basis each column represents a basis vector

  Returns:
    Nothing.
  """

    plt.figure(figsize=[4, 4])
    plt.plot(X[:, 0], X[:, 1], '.', color=[.5, .5, .5], label='Data')
    plt.axis('equal')
    plt.xlabel('Neuron 1 activity')
    plt.ylabel('Neuron 2 activity')
    plt.plot([0, W[0, 0]], [0, W[1, 0]], color='r', linewidth=3,
             label='Basis vector 1')
    plt.plot([0, W[0, 1]], [0, W[1, 1]], color='b', linewidth=3,
             label='Basis vector 2')
    plt.legend()
    plt.show()


def plot_data_new_basis(Y):
    """
  Plots bivariate data after transformation to new bases.
  Similar to plot_data but with colors corresponding to projections onto
  basis 1 (red) and basis 2 (blue). The title indicates the sample correlation
  calculated from the data.

  Note that samples are re-sorted in ascending order for the first
  random variable.

  Args:
    Y (numpy array of floats) : Data matrix in new basis each column
                                corresponds to a different random variable

  Returns:
    Nothing.
  """
    fig = plt.figure(figsize=[8, 4])
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(Y[:, 0], 'r')
    plt.xlabel
    plt.ylabel('Projection \n basis vector 1')
    plt.title(f'Sample var 1: {np.var(Y[:, 0]):.1f}')
    ax1.set_xticklabels([])
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(Y[:, 1], 'b')
    plt.xlabel('Sample number')
    plt.ylabel('Projection \n basis vector 2')
    plt.title(f'Sample var 2: {np.var(Y[:, 1]):.1f}')
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.plot(Y[:, 0], Y[:, 1], '.', color=[.5, .5, .5])
    ax3.axis('equal')
    plt.xlabel('Projection basis vector 1')
    plt.ylabel('Projection basis vector 2')
    plt.title(f'Sample corr: {np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]:.1f}')
    plt.show()


# @title Plotting Functions

def plot_eigenvalues(evals, xlimit=False):
    """
  Plots eigenvalues.

  Args:
     (numpy array of floats) : Vector of eigenvalues
     (boolean) : enable plt.show()
  Returns:
    Nothing.

  """

    plt.figure()
    plt.plot(np.arange(1, len(evals) + 1), evals, 'o-k')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title('Scree plot')
    if xlimit:
        plt.xlim([0, 100])  # limit x-axis up to 100 for zooming
    plt.show()


def plot_data(X):
    """
  Plots bivariate data. Includes a plot of each random variable, and a scatter
  scatter plot of their joint activity. The title indicates the sample
  correlation calculated from the data.

  Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable

  Returns:
    Nothing.
  """

    fig = plt.figure(figsize=[8, 4])
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(X[:, 0], color='k')
    plt.ylabel('Neuron 1')
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(X[:, 1], color='k')
    plt.xlabel('Sample Number (sorted)')
    plt.ylabel('Neuron 2')
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.plot(X[:, 0], X[:, 1], '.', markerfacecolor=[.5, .5, .5],
             markeredgewidth=0)
    ax3.axis('equal')
    plt.xlabel('Neuron 1 activity')
    plt.ylabel('Neuron 2 activity')
    plt.title('Sample corr: {:.1f}'.format(np.corrcoef(X[:, 0], X[:, 1])[0, 1]))
    plt.show()


def plot_data_new_basis(Y):
    """
  Plots bivariate data after transformation to new bases. Similar to plot_data
  but with colors corresponding to projections onto basis 1 (red) and
  basis 2 (blue).
  The title indicates the sample correlation calculated from the data.

  Note that samples are re-sorted in ascending order for the first random
  variable.

  Args:
    Y (numpy array of floats) : Data matrix in new basis each column
                                corresponds to a different random variable

  Returns:
    Nothing.
  """

    fig = plt.figure(figsize=[8, 4])
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(Y[:, 0], 'r')
    plt.ylabel('Projection \n basis vector 1')
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(Y[:, 1], 'b')
    plt.xlabel('Sample number')
    plt.ylabel('Projection \n basis vector 2')
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.plot(Y[:, 0], Y[:, 1], '.', color=[.5, .5, .5])
    ax3.axis('equal')
    plt.xlabel('Projection basis vector 1')
    plt.ylabel('Projection basis vector 2')
    plt.title('Sample corr: {:.1f}'.format(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]))
    plt.show()


def plot_basis_vectors(X, W):
    """
  Plots bivariate data as well as new basis vectors.

  Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable
    W (numpy array of floats) : Square matrix representing new orthonormal
                                basis each column represents a basis vector

  Returns:
    Nothing.
  """

    plt.figure(figsize=[4, 4])
    plt.plot(X[:, 0], X[:, 1], '.', color=[.5, .5, .5], label='Data')
    plt.axis('equal')
    plt.xlabel('Neuron 1 activity')
    plt.ylabel('Neuron 2 activity')
    plt.plot([0, W[0, 0]], [0, W[1, 0]], color='r', linewidth=3,
             label='Basis vector 1')
    plt.plot([0, W[0, 1]], [0, W[1, 1]], color='b', linewidth=3,
             label='Basis vector 2')
    plt.legend()
    plt.show()


# @title Helper functions

def sort_evals_descending(evals, evectors):
    """
  Sorts eigenvalues and eigenvectors in decreasing order. Also aligns first two
  eigenvectors to be in first two quadrants (if 2D).

  Args:
    evals (numpy array of floats)    : Vector of eigenvalues
    evectors (numpy array of floats) : Corresponding matrix of eigenvectors
                                        each column corresponds to a different
                                        eigenvalue

  Returns:
    (numpy array of floats)          : Vector of eigenvalues after sorting
    (numpy array of floats)          : Matrix of eigenvectors after sorting
  """

    index = np.flip(np.argsort(evals))
    evals = evals[index]
    evectors = evectors[:, index]
    if evals.shape[0] == 2:
        if np.arccos(np.matmul(evectors[:, 0],
                               1 / np.sqrt(2) * np.array([1, 1]))) > np.pi / 2:
            evectors[:, 0] = -evectors[:, 0]
        if np.arccos(np.matmul(evectors[:, 1],
                               1 / np.sqrt(2) * np.array([-1, 1]))) > np.pi / 2:
            evectors[:, 1] = -evectors[:, 1]
    return evals, evectors


def get_data(cov_matrix):
    """
  Returns a matrix of 1000 samples from a bivariate, zero-mean Gaussian

  Note that samples are sorted in ascending order for the first random
  variable.

  Args:
    var_1 (scalar)                     : variance of the first random variable
    var_2 (scalar)                     : variance of the second random variable
    cov_matrix (numpy array of floats) : desired covariance matrix

  Returns:
    (numpy array of floats)            : samples from the bivariate Gaussian,
                                          with each column corresponding to a
                                          different random variable
  """

    mean = np.array([0, 0])
    X = np.random.multivariate_normal(mean, cov_matrix, size=1000)
    indices_for_sorting = np.argsort(X[:, 0])
    X = X[indices_for_sorting, :]
    return X


def calculate_cov_matrix(var_1, var_2, corr_coef):
    """
  Calculates the covariance matrix based on the variances and
  correlation coefficient.

  Args:
    var_1 (scalar)         :  variance of the first random variable
    var_2 (scalar)         :  variance of the second random variable
    corr_coef (scalar)     :  correlation coefficient

  Returns:
    (numpy array of floats) : covariance matrix
  """
    cov = corr_coef * np.sqrt(var_1 * var_2)
    cov_matrix = np.array([[var_1, cov], [cov, var_2]])
    return cov_matrix


def define_orthonormal_basis(u):
    """
  Calculates an orthonormal basis given an arbitrary vector u.

  Args:
    u (numpy array of floats) : arbitrary 2D vector used for new basis

  Returns:
    (numpy array of floats)   : new orthonormal basis columns correspond to
                                basis vectors
  """

    u = u / np.sqrt(u[0] ** 2 + u[1] ** 2)
    w = np.array([-u[1], u[0]])
    W = np.column_stack((u, w))
    return W


def change_of_basis(X, W):
    """
  Projects data onto a new basis.

  Args:
    X (numpy array of floats) : Data matrix each column corresponding to a
                                different random variable
    W (numpy array of floats) : new orthonormal basis columns correspond to
                                basis vectors

  Returns:
    (numpy array of floats)   : Data matrix expressed in new basis
  """

    Y = np.matmul(X, W)
    return Y


# @title Plotting Functions

def plot_variance_explained(variance_explained):
    """
  Plots eigenvalues.

  Args:
    variance_explained (numpy array of floats) : Vector of variance explained
                                                 for each PC

  Returns:
    Nothing.

  """

    plt.figure()
    plt.plot(np.arange(1, len(variance_explained) + 1), variance_explained,
             '--k')
    plt.xlabel('Number of components')
    plt.ylabel('Variance explained')
    plt.show()


def plot_MNIST_reconstruction(X, X_reconstructed, keep_dims):
    """
  Plots 9 images in the MNIST dataset side-by-side with the reconstructed
  images.

  Args:
    X (numpy array of floats)               : Data matrix each column
                                              corresponds to a different
                                              random variable
    X_reconstructed (numpy array of floats) : Data matrix each column
                                              corresponds to a different
                                              random variable
    keep_dims (int)                         : Dimensions to keep

  Returns:
    Nothing.
  """

    plt.figure()
    ax = plt.subplot(121)
    k = 0
    for k1 in range(3):
        for k2 in range(3):
            k = k + 1
            plt.imshow(np.reshape(X[k, :], (28, 28)),
                       extent=[(k1 + 1) * 28, k1 * 28, (k2 + 1) * 28, k2 * 28],
                       vmin=0, vmax=255)
    plt.xlim((3 * 28, 0))
    plt.ylim((3 * 28, 0))
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title('Data')
    plt.clim([0, 250])
    ax = plt.subplot(122)
    k = 0
    for k1 in range(3):
        for k2 in range(3):
            k = k + 1
            plt.imshow(np.reshape(np.real(X_reconstructed[k, :]), (28, 28)),
                       extent=[(k1 + 1) * 28, k1 * 28, (k2 + 1) * 28, k2 * 28],
                       vmin=0, vmax=255)
    plt.xlim((3 * 28, 0))
    plt.ylim((3 * 28, 0))
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.clim([0, 250])
    plt.title(f'Reconstructed K: {keep_dims}')
    plt.tight_layout()
    plt.show()


def plot_MNIST_sample(X):
    """
  Plots 9 images in the MNIST dataset.

  Args:
     X (numpy array of floats) : Data matrix each column corresponds to a
                                 different random variable

  Returns:
    Nothing.

  """

    fig, ax = plt.subplots()
    k = 0
    for k1 in range(3):
        for k2 in range(3):
            k = k + 1
            plt.imshow(np.reshape(X[k, :], (28, 28)),
                       extent=[(k1 + 1) * 28, k1 * 28, (k2 + 1) * 28, k2 * 28],
                       vmin=0, vmax=255)
    plt.xlim((3 * 28, 0))
    plt.ylim((3 * 28, 0))
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False)
    plt.clim([0, 250])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def plot_MNIST_weights(weights):
    """
  Visualize PCA basis vector weights for MNIST. Red = positive weights,
  blue = negative weights, white = zero weight.

  Args:
     weights (numpy array of floats) : PCA basis vector

  Returns:
     Nothing.
  """

    fig, ax = plt.subplots()
    plt.imshow(np.real(np.reshape(weights, (28, 28))), cmap='seismic')
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False)
    plt.clim(-.15, .15)
    plt.colorbar(ticks=[-.15, -.1, -.05, 0, .05, .1, .15])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


# @title Helper Functions

def add_noise(X, frac_noisy_pixels):
    """
  Randomly corrupts a fraction of the pixels by setting them to random values.

  Args:
     X (numpy array of floats)  : Data matrix
     frac_noisy_pixels (scalar) : Fraction of noisy pixels

  Returns:
     (numpy array of floats)    : Data matrix + noise

  """

    X_noisy = np.reshape(X, (X.shape[0] * X.shape[1]))
    N_noise_ixs = int(X_noisy.shape[0] * frac_noisy_pixels)
    noise_ixs = np.random.choice(X_noisy.shape[0], size=N_noise_ixs,
                                 replace=False)
    X_noisy[noise_ixs] = np.random.uniform(0, 255, noise_ixs.shape)
    X_noisy = np.reshape(X_noisy, (X.shape[0], X.shape[1]))

    return X_noisy


def get_sample_cov_matrix(X):
    """
  Returns the sample covariance matrix of data X.

  Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable

  Returns:
    (numpy array of floats)   : Covariance matrix
"""

    X = X - np.mean(X, 0)
    cov_matrix = 1 / X.shape[0] * np.matmul(X.T, X)
    return cov_matrix


def pca(X):
    """
  Performs PCA on multivariate data. Eigenvalues are sorted in decreasing order

  Args:
     X (numpy array of floats) :   Data matrix each column corresponds to a
                                   different random variable

  Returns:
    (numpy array of floats)    : Data projected onto the new basis
    (numpy array of floats)    : Corresponding matrix of eigenvectors
    (numpy array of floats)    : Vector of eigenvalues

  """

    X = X - np.mean(X, 0)
    cov_matrix = get_sample_cov_matrix(X)
    evals, evectors = np.linalg.eigh(cov_matrix)
    evals, evectors = sort_evals_descending(evals, evectors)
    score = change_of_basis(X, evectors)

    return score, evectors, evals


# @title Plotting Functions

def visualize_components(component1, component2, labels, show=True):
    """
  Plots a 2D representation of the data for visualization with categories
  labelled as different colors.

  Args:
    component1 (numpy array of floats) : Vector of component 1 scores
    component2 (numpy array of floats) : Vector of component 2 scores
    labels (numpy array of floats)     : Vector corresponding to categories of
                                         samples

  Returns:
    Nothing.

  """

    plt.figure()
    plt.scatter(x=component1, y=component2, c=labels, cmap='tab10')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    if show:
        plt.show()
