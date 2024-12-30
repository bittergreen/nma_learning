# Imports
import os
import numpy as np

import torch
from torch import nn
from torch import optim

import matplotlib as mpl
from matplotlib import pyplot as plt

from w1.d5_deeplearning.utils import get_data, load_data, fname, plot_train_loss


class DeepNetReLU(nn.Module):

    def __init__(self, n_input, n_hidden):
        super().__init__()
        self.input_layer = nn.Linear(n_input, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 1)

    def forward(self, r):
        h = torch.relu(self.input_layer(r))
        o = self.output_layer(h)
        return o


def train(net, loss_fn, train_data, train_labels, n_epochs=50, learning_rate=1e-4):
    """Run gradient descent to optimize parameters of a given network

      Args:
        net (nn.Module): PyTorch network whose parameters to optimize
        loss_fn: built-in PyTorch loss function to minimize
        train_data (torch.Tensor): n_train x n_neurons tensor with neural
          responses to train on
        train_labels (torch.Tensor): n_train x 1 tensor with orientations of the
          stimuli corresponding to each row of train_data
        n_epochs (int, optional): number of epochs of gradient descent to run
        learning_rate (float, optional): learning rate to use for gradient descent

      Returns:
        (list): training loss over iterations

    """

    # Initialize PyTorch SGD optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # Placeholder to save the loss at each iteration
    train_loss = []

    # Loop over epochs
    for i in range(n_epochs):
        # compute network output from inputs in train_data
        out = net(train_data)  # compute network output from inputs in train_data

        # evaluate loss function
        loss = loss_fn(out, train_labels)

        # Clear previous gradients
        optimizer.zero_grad()

        # Compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        # Store current value of loss
        train_loss.append(loss.item())  # .item() needed to transform the tensor output of loss_fn to a scalar

        # Track progress
        if (i + 1) % (n_epochs // 5) == 0:
            print(f'iteration {i + 1}/{n_epochs} | loss: {loss.item():.3f}')

    return train_loss


if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(4)
    torch.manual_seed(4)

    # Load data
    resp_all, stimuli_all = load_data(fname)  # argument to this function specifies bin width
    n_stimuli, n_neurons = resp_all.shape

    # Split data into training set and testing set
    n_train = int(0.6 * n_stimuli)  # use 60% of all data for training set
    ishuffle = torch.randperm(n_stimuli)
    itrain = ishuffle[:n_train]  # indices of data samples to include in training set
    itest = ishuffle[n_train:]  # indices of data samples to include in testing set
    stimuli_test = stimuli_all[itest]
    resp_test = resp_all[itest]
    stimuli_train = stimuli_all[itrain]
    resp_train = resp_all[itrain]

    # Set random seeds for reproducibility
    np.random.seed(1)
    torch.manual_seed(1)

    # Initialize network with 10 hidden units
    net = DeepNetReLU(n_neurons, 10)

    # Initialize built-in PyTorch MSE loss function
    loss_fn = nn.MSELoss()

    # Run gradient descent on data
    train_loss = train(net, loss_fn, resp_train, stimuli_train)

    # Plot the training loss over iterations of GD
    plot_train_loss(train_loss)

