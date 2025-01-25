# Imports
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from compneuro.w1.d3_glm.utils import load_steinmetz_data, plot_weights, plot_non_zero_coefs, plot_model_selection


def sigmoid(z):
    """Return the logistic transform of z."""
    sigmoid = 1 / (1 + np.exp(-z))

    return sigmoid


def build_model(X, y):
    log_reg = LogisticRegression(penalty=None)
    log_reg.fit(X, y)
    return log_reg


def compute_accuracy(X, y, model):
    """Compute accuracy of classifier predictions.

      Args:
        X (2D array): Data matrix
        y (1D array): Label vector
        model (sklearn estimator): Classifier with trained weights.

      Returns:
        accuracy (float): Proportion of correct predictions.
    """
    y_pred = model.predict(X)

    accuracy = (y == y_pred).sum() / len(y)

    return accuracy


def compare_models(X, y):
    log_reg = LogisticRegression(penalty=None).fit(X, y)
    log_reg_l2 = LogisticRegression(penalty="l2", C=1).fit(X, y)
    log_reg_l1 = LogisticRegression(penalty="l1", C=1, solver="saga", max_iter=5000)
    log_reg_l1.fit(X, y)
    # now show the two models
    models = {
        "No regularization": log_reg,
        "$L_2$ (C = 1)": log_reg_l2,
        "$L_1$ (C = 1)": log_reg_l1,
    }
    plot_weights(models, sharey=False)


def count_non_zero_coefs(X, y, C_values):
    """Fit models with different L1 penalty values and count non-zero coefficients.

      Args:
        X (2D array): Data matrix
        y (1D array): Label vector
        C_values (1D array): List of hyperparameter values

      Returns:
        non_zero_coefs (list): number of coefficients in each model that are nonzero

    """
    non_zero_coefs = []
    for C in C_values:
        # Initialize and fit the model
        # (Hint, you may need to set max_iter)
        model = LogisticRegression(penalty='l1', C=C, solver="saga", max_iter=5000)
        model.fit(X, y)

        # Get the coefs of the fit model (in sklearn, we can do this using model.coef_)
        coefs = model.coef_

        # Count the number of non-zero elements in coefs
        non_zero = np.sum(coefs != 0)
        non_zero_coefs.append(non_zero)

    return non_zero_coefs


def plot_l1_C(X, y):
    # Use log-spaced values for C
    C_values = np.logspace(-4, 4, 5)

    # Count non zero coefficients
    non_zero_l1 = count_non_zero_coefs(X, y, C_values)

    # Visualize
    plot_non_zero_coefs(C_values, non_zero_l1, n_voxels=X.shape[1])


def model_selection(X, y, C_values):
    """Compute CV accuracy for each C value.

      Args:
        X (2D array): Data matrix
        y (1D array): Label vector
        C_values (1D array): Array of hyperparameter values

      Returns:
        accuracies (1D array): CV accuracy with each value of C

    """
    accuracies = []
    for C in C_values:
        # Initialize and fit the model
        # (Hint, you may need to set max_iter)
        model = LogisticRegression(penalty='l1', C=C, solver="saga", max_iter=5000)

        # Get the accuracy for each test split using cross-validation
        accs = cross_val_score(model, X, y, cv=8)

        # Store the average test accuracy for this value of C
        accuracies.append(accs.mean())

    return accuracies


if __name__ == '__main__':
    data = load_steinmetz_data()
    X = data["spikes"]
    y = data["choices"]
    # Visualize
    # plot_function(sigmoid, "\sigma", "z", (-10, 10))
    # Compute train accuracy
    model = build_model(X, y)
    train_accuracy = compute_accuracy(X, y, model)
    print(f"Accuracy on the training data: {train_accuracy:.2%}")

    accuracies = cross_val_score(LogisticRegression(penalty=None), X, y, cv=8)  # k=8 cross validation
    print(accuracies.mean())

    # compare_models(X, y)
    # plot_l1_C(X, y)

    # Use log-spaced values for C
    C_values = np.logspace(-4, 4, 9)

    # Compute accuracies
    accuracies = model_selection(X, y, C_values)

    # Visualize
    plot_model_selection(C_values, accuracies)

