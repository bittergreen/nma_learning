# Imports
import numpy as np
from scipy.stats import zscore
import matplotlib as mpl
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from w1.d5_deeplearning.utils import CNN, grating, train, load_data, fname, get_hidden_activity, plot_multiple_rdm, \
    plot_rdm_rdm_correlations

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


def RDM(resp):
    """Compute the representational dissimilarity matrix (RDM)

      Args:
        resp (ndarray): S x N matrix with population responses to
          each stimulus in each row

      Returns:
        ndarray: S x S representational dissimilarity matrix
      """

    # z-score responses to each stimulus
    zresp = zscore(resp, axis=1)

    # Compute RDM
    RDM = 1 - (zresp @ zresp.T) / zresp.shape[1]

    return RDM


def correlate_rdms(rdm1, rdm2):
    """Correlate off-diagonal elements of two RDM's

      Args:
        rdm1 (np.ndarray): S x S representational dissimilarity matrix
        rdm2 (np.ndarray): S x S representational dissimilarity matrix to
          correlate with rdm1

      Returns:
        float: correlation coefficient between the off-diagonal elements
          of rdm1 and rdm2

      """

    # Extract off-diagonal elements of each RDM
    ioffdiag = np.triu_indices(rdm1.shape[0], k=1)  # indices of off-diagonal elements
    rdm1_offdiag = rdm1[ioffdiag]
    rdm2_offdiag = rdm2[ioffdiag]

    corr_coef = np.corrcoef(rdm1_offdiag, rdm2_offdiag)[0,1]

    return corr_coef


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

    # Compute RDMs for each layer
    rdm_dict = {label: RDM(resp) for label, resp in resp_dict.items()}

    # Plot RDMs
    plot_multiple_rdm(rdm_dict, resp_dict)

    # Split RDMs into V1 responses and model responses
    rdm_model = rdm_dict.copy()
    rdm_v1 = rdm_model.pop('V1 data')

    # Correlate off-diagonal terms of dissimilarity matrices
    rdm_sim = {label: correlate_rdms(rdm_v1, rdm) for label, rdm in rdm_model.items()}

    # Visualize
    plot_rdm_rdm_correlations(rdm_sim)
