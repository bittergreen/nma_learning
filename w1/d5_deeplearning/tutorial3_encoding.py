# Imports
import numpy as np
from scipy.stats import zscore
import matplotlib as mpl
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from w1.d5_deeplearning.utils import CNN, grating, train, load_data, fname, get_hidden_activity

# Set random seeds for reproducibility
np.random.seed(12)
torch.manual_seed(12)


def run_training_trial():
    h, w = grating(0).shape  # height and width of stimulus

    # Initialize CNN model
    net = CNN(h, w)

    # Build training set to train it on
    n_train = 1000  # size of training set

    # sample n_train random orientations between -90 and +90 degrees
    ori = (np.random.rand(n_train) - 0.5) * 180

    # build orientated grating stimuli
    stimuli = torch.stack([grating(i) for i in ori])

    # stimulus tilt: 1. if tilted right, 0. if tilted left, as a column vector
    tilt = torch.tensor(ori > 0).type(torch.float).unsqueeze(-1)

    # Train model
    train(net, stimuli, tilt)

    return net


if __name__ == '__main__':

    # Initialize CNN model
    net = run_training_trial()

    # Load mouse V1 data
    resp_v1, ori = load_data(fname)

    # Extract model internal representations of each stimulus in the V1 data
    # construct grating stimuli for each orientation presented in the V1 data
    stimuli = torch.stack([grating(a.item()) for a in ori])
    layer_labels = ['pool', 'fc']
    resp_model = get_hidden_activity(net, stimuli, layer_labels)

    # Aggregate all responses into one dict
    resp_dict = {}
    resp_dict['V1 data'] = resp_v1
    for k, v in resp_model.items():
        label = f"model\n'{k}' layer"
        resp_dict[label] = v

