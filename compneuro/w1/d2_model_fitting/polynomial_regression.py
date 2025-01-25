import numpy as np
from matplotlib import pyplot as plt

from compneuro.w1.d2_model_fitting.utils import evaluate_fits


def plot_noisy_samples():
    np.random.seed(1234)

    # Set parameters
    theta = [0, -2, -3]
    n_samples = 40

    # Draw x and calculate y
    n_regressors = len(theta)
    x0 = np.ones((n_samples, 1))
    x1 = np.random.uniform(-2, 2, (n_samples, 1))
    x2 = np.random.uniform(-2, 2, (n_samples, 1))
    X = np.hstack((x0, x1, x2))
    noise = np.random.randn(n_samples)
    y = X @ theta + noise

    ax = plt.subplot(projection='3d')
    ax.plot(X[:, 1], X[:, 2], y, '.')

    ax.set(
        xlabel='$x_1$',
        ylabel='$x_2$',
        zlabel='y'
    )
    plt.tight_layout()
    # plt.show()
    return X, y


def ordinary_least_squares(X, y):
    """Ordinary least squares estimator for linear regression.

      Args:
        x (ndarray): design matrix of shape (n_samples, n_regressors)
        y (ndarray): vector of measurements of shape (n_samples)

      Returns:
        ndarray: estimated parameter values of shape (n_regressors)
    """
    # Compute theta_hat using OLS
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    return theta_hat


def visualize_multiple_lr_mse():
    # @markdown Execute this cell to visualize data and predicted plane

    theta_hat = ordinary_least_squares(X, y)
    xx, yy = np.mgrid[-2:2:50j, -2:2:50j]
    y_hat_grid = np.array([xx.flatten(), yy.flatten()]).T @ theta_hat[1:]
    y_hat_grid = y_hat_grid.reshape((50, 50))

    ax = plt.subplot(projection='3d')
    ax.plot(X[:, 1], X[:, 2], y, '.')
    ax.plot_surface(xx, yy, y_hat_grid, linewidth=0, alpha=0.5, color='C1',
                    cmap=plt.get_cmap('coolwarm'))

    for i in range(len(X)):
        ax.plot((X[i, 1], X[i, 1]),
                (X[i, 2], X[i, 2]),
                (y[i], y_hat[i]),
                'g-', alpha=.5)

    ax.set(
        xlabel='$x_1$',
        ylabel='$x_2$',
        zlabel='y'
    )
    plt.tight_layout()
    plt.show()


def plot_polynomial_samples_with_noise():
    # @markdown Execute this cell to simulate some data

    # setting a fixed seed to our random number generator ensures we will always
    # get the same psuedorandom number sequence
    np.random.seed(121)
    n_samples = 30
    x = np.random.uniform(-2, 2.5, n_samples)  # inputs uniformly sampled from [-2, 2.5)
    y = x ** 2 - x - 2  # computing the outputs

    output_noise = 1 / 8 * np.random.randn(n_samples)
    y += output_noise  # adding some output noise

    input_noise = 1 / 2 * np.random.randn(n_samples)
    x += input_noise  # adding some input noise

    fig, ax = plt.subplots()
    ax.scatter(x, y)  # produces a scatter plot
    ax.set(xlabel='x', ylabel='y')
    return x, y


def make_design_matrix(x, order):
    """Create the design matrix of inputs for use in polynomial regression

      Args:
        x (ndarray): input vector of shape (n_samples)
        order (scalar): polynomial regression order

      Returns:
        ndarray: design matrix for polynomial regression of shape (samples, order+1)
    """
    # Broadcast to shape (n x 1) so dimensions work
    if x.ndim == 1:
        x = x[:, None]

    # if x has more than one feature, we don't want multiple columns of ones so we assign
    # x^0 here
    design_matrix = np.ones((x.shape[0], 1))

    # Loop through rest of degrees and stack columns (hint: np.hstack)
    for degree in range(1, order + 1):
        design_matrix = np.hstack([design_matrix, x ** degree])

    return design_matrix


def plot_fitted_polynomials(x, y, theta_hat, max_order):
    """ Plot polynomials of different orders

      Args:
        x (ndarray): input vector of shape (n_samples)
        y (ndarray): vector of measurements of shape (n_samples)
        theta_hat (dict): polynomial regression weights for different orders
    """

    x_grid = np.linspace(x.min() - .5, x.max() + .5)

    plt.figure()

    for order in range(0, max_order + 1):
        X_design = make_design_matrix(x_grid, order)
        plt.plot(x_grid, X_design @ theta_hat[order])

    plt.ylabel('y')
    plt.xlabel('x')
    plt.plot(x, y, 'C0.')
    plt.legend([f'order {o}' for o in range(max_order + 1)], loc=1)
    plt.title('polynomial fits')
    plt.show()


def solve_poly_reg(x, y, max_order):
    """Fit a polynomial regression model for each order 0 through max_order.

      Args:
        x (ndarray): input vector of shape (n_samples)
        y (ndarray): vector of measurements of shape (n_samples)
        max_order (scalar): max order for polynomial fits

      Returns:
        dict: fitted weights for each polynomial model (dict key is order)
    """

    # Create a dictionary with polynomial order as keys,
    # and np array of theta_hat (weights) as the values
    theta_hats = {}

    # Loop over polynomial orders from 0 through max_order
    for order in range(max_order + 1):

        # Create design matrix
        X_design = make_design_matrix(x, order)

        # Fit polynomial model
        this_theta = ordinary_least_squares(X_design, y)

        theta_hats[order] = this_theta

    return theta_hats


def evaluate(x, y, max_order, theta_hats):
    mse_list = []
    order_list = list(range(max_order + 1))

    for order in order_list:
        X_design = make_design_matrix(x, order)

        # Get prediction for the polynomial regression model of this order
        y_hat = X_design @ theta_hats[order]

        # Compute the residuals
        residuals = (y - y_hat)

        # Compute the MSE
        mse = np.mean(residuals ** 2)

        mse_list.append(mse)

    # Visualize MSE of fits
    evaluate_fits(order_list, mse_list)


if __name__ == '__main__':
    X, y = plot_noisy_samples()
    theta_hat = ordinary_least_squares(X, y)
    print(theta_hat)
    y_hat = X @ theta_hat
    # Compute MSE
    print(f"MSE = {np.mean((y - y_hat) ** 2):.2f}")
    # visualize_multiple_lr_mse()
    x, y = plot_polynomial_samples_with_noise()
    order = 5
    X_design = make_design_matrix(x, order)

    print(X_design[0:2, 0:2])

    max_order = 5
    theta_hats = solve_poly_reg(x, y, max_order)

    # Visualize
    # plot_fitted_polynomials(x, y, theta_hats, max_order)

    evaluate(x, y, max_order, theta_hats)

