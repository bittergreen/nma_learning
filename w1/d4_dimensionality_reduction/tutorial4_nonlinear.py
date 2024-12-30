# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

from utils import visualize_components


if __name__ == '__main__':
    # Get images
    mnist = fetch_openml(name='mnist_784', as_frame=False, parser='auto')
    X_all = mnist.data

    # Get labels
    labels_all = np.array([int(k) for k in mnist.target])

    # Initializes PCA
    pca_model = PCA(n_components=2)

    # Performs PCA
    pca_model.fit(X_all)

    # Take only the first 2000 samples with the corresponding labels
    X, labels = X_all[:2000], labels_all[:2000]

    # Perform PCA
    scores = pca_model.transform(X)

    # Plot the data and reconstruction
    visualize_components(scores[:, 0], scores[:, 1], labels)

