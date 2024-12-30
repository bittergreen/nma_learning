# Imports
import os
import numpy as np

import torch
from torch import nn
from torch import optim

import matplotlib as mpl
from matplotlib import pyplot as plt

# @title Data retrieval and loading
import hashlib
import requests

fname = "W3D4_stringer_oribinned1.npz"
url = "https://osf.io/683xc/download"
expected_md5 = "436599dfd8ebe6019f066c38aed20580"


def download_data():
    if not os.path.isfile(fname):
        try:
            r = requests.get(url)
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("!!! Failed to download data !!!")
            elif hashlib.md5(r.content).hexdigest() != expected_md5:
                print("!!! Data download appears corrupted !!!")
            else:
                with open(fname, "wb") as fid:
                    fid.write(r.content)


# @title Plotting Functions

def plot_data_matrix(X, ax, show=False):
    """Visualize data matrix of neural responses using a heatmap

  Args:
    X (torch.Tensor or np.ndarray): matrix of neural responses to visualize
        with a heatmap
    ax (matplotlib axes): where to plot
    show (boolean): enable plt.show()

  """

    cax = ax.imshow(X, cmap=mpl.cm.pink, vmin=np.percentile(X, 1),
                    vmax=np.percentile(X, 99))
    cbar = plt.colorbar(cax, ax=ax, label='normalized neural response')

    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    if show:
        plt.show()


def plot_train_loss(train_loss):
    plt.plot(train_loss)
    plt.xlim([0, None])
    plt.ylim([0, None])
    plt.xlabel('iterations of gradient descent')
    plt.ylabel('mean squared error')
    plt.show()


# @title Helper Functions

def load_data(data_name, bin_width=1):
    """Load mouse V1 data from Stringer et al. (2019)

  Data from study reported in this preprint:
  https://www.biorxiv.org/content/10.1101/679324v2.abstract

  These data comprise time-averaged responses of ~20,000 neurons
  to ~4,000 stimulus gratings of different orientations, recorded
  through Calcium imaging. The responses have been normalized by
  spontaneous levels of activity and then z-scored over stimuli, so
  expect negative numbers. They have also been binned and averaged
  to each degree of orientation.

  This function returns the relevant data (neural responses and
  stimulus orientations) in a torch.Tensor of data type torch.float32
  in order to match the default data type for nn.Parameters in
  Google Colab.

  This function will actually average responses to stimuli with orientations
  falling within bins specified by the bin_width argument. This helps
  produce individual neural "responses" with smoother and more
  interpretable tuning curves.

  Args:
    bin_width (float): size of stimulus bins over which to average neural
      responses

  Returns:
    resp (torch.Tensor): n_stimuli x n_neurons matrix of neural responses,
        each row contains the responses of each neuron to a given stimulus.
        As mentioned above, neural "response" is actually an average over
        responses to stimuli with similar angles falling within specified bins.
    stimuli: (torch.Tensor): n_stimuli x 1 column vector with orientation
        of each stimulus, in degrees. This is actually the mean orientation
        of all stimuli in each bin.

  """
    with np.load(data_name) as dobj:
        data = dict(**dobj)
    resp = data['resp']
    stimuli = data['stimuli']

    if bin_width > 1:
        # Bin neural responses and stimuli
        bins = np.digitize(stimuli, np.arange(0, 360 + bin_width, bin_width))
        stimuli_binned = np.array([stimuli[bins == i].mean() for i in np.unique(bins)])
        resp_binned = np.array([resp[bins == i, :].mean(0) for i in np.unique(bins)])
    else:
        resp_binned = resp
        stimuli_binned = stimuli

    # Return as torch.Tensor
    resp_tensor = torch.tensor(resp_binned, dtype=torch.float32)
    stimuli_tensor = torch.tensor(stimuli_binned, dtype=torch.float32).unsqueeze(
        1)  # add singleton dimension to make a column vector

    return resp_tensor, stimuli_tensor


def get_data(n_stim, train_data, train_labels):
    """ Return n_stim randomly drawn stimuli/resp pairs

      Args:
        n_stim (scalar): number of stimuli to draw
        resp (torch.Tensor):
        train_data (torch.Tensor): n_train x n_neurons tensor with neural
          responses to train on
        train_labels (torch.Tensor): n_train x 1 tensor with orientations of the
          stimuli corresponding to each row of train_data, in radians

      Returns:
        (torch.Tensor, torch.Tensor): n_stim x n_neurons tensor of neural responses and n_stim x 1 of orientations respectively
    """
    n_stimuli = train_labels.shape[0]
    istim = np.random.choice(n_stimuli, n_stim)
    r = train_data[istim]  # neural responses to this stimulus
    ori = train_labels[istim]  # true stimulus orientation

    return r, ori


def real_get_data():
    download_data()
    # @markdown Execute this cell to load and visualize data

    # Load data
    resp_all, stimuli_all = load_data(fname)  # argument to this function specifies bin width
    n_stimuli, n_neurons = resp_all.shape

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 6, 5))

    # Visualize data matrix
    plot_data_matrix(resp_all[:, :100].T, ax1)  # plot responses of first 100 neurons
    ax1.set_xlabel('stimulus')
    ax1.set_ylabel('neuron')

    # Plot tuning curves of three random neurons
    ineurons = np.random.choice(n_neurons, 3, replace=False)  # pick three random neurons
    ax2.plot(stimuli_all, resp_all[:, ineurons])
    ax2.set_xlabel('stimulus orientation ($^o$)')
    ax2.set_ylabel('neural response')
    ax2.set_xticks(np.linspace(0, 360, 5))

    fig.suptitle(f'{n_neurons} neurons in response to {n_stimuli} stimuli')
    fig.tight_layout()
    plt.show()
    """

    # @markdown Execute this cell to split into training and test sets

    # Set random seeds for reproducibility
    np.random.seed(4)
    torch.manual_seed(4)

    # Split data into training set and testing set
    n_train = int(0.6 * n_stimuli)  # use 60% of all data for training set
    ishuffle = torch.randperm(n_stimuli)
    itrain = ishuffle[:n_train]  # indices of data samples to include in training set
    itest = ishuffle[n_train:]  # indices of data samples to include in testing set
    stimuli_test = stimuli_all[itest]
    resp_test = resp_all[itest]
    stimuli_train = stimuli_all[itrain]
    resp_train = resp_all[itrain]

    # Get neural responses (r) to and orientation (ori) to one stimulus in dataset
    r, ori = get_data(1, resp_train, stimuli_train)  # using helper function get_data

    return n_neurons, r, ori


# @title Plotting Functions

def show_stimulus(img, ax=None, show=False):
    """Visualize a stimulus"""
    if ax is None:
        ax = plt.gca()
    ax.imshow(img + 0.5, cmap=mpl.cm.binary)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if show:
        plt.show()


def plot_weights(weights, channels=[0]):
    """ plot convolutional channel weights
  Args:
      weights: weights of convolutional filters (conv_channels x K x K)
      channels: which conv channels to plot
  """
    wmax = torch.abs(weights).max()
    fig, axs = plt.subplots(1, len(channels), figsize=(12, 2.5))
    for i, channel in enumerate(channels):
        im = axs[i].imshow(weights[channel, 0], vmin=-wmax, vmax=wmax, cmap='bwr')
        axs[i].set_title(f'channel {channel}')

        cb_ax = fig.add_axes([1, 0.1, 0.05, 0.8])
        plt.colorbar(im, ax=cb_ax)
        cb_ax.axis('off')
    plt.show()


def plot_example_activations(stimuli, act, channels=[0]):
    """ plot activations act and corresponding stimulus
  Args:
    stimuli: stimulus input to convolutional layer (n x h x w) or (h x w)
    act: activations of convolutional layer (n_bins x conv_channels x n_bins)
    channels: which conv channels to plot
  """
    if stimuli.ndim > 2:
        n_stimuli = stimuli.shape[0]
    else:
        stimuli = stimuli.unsqueeze(0)
        n_stimuli = 1

    fig, axs = plt.subplots(n_stimuli, 1 + len(channels), figsize=(12, 12))

    # plot stimulus
    for i in range(n_stimuli):
        show_stimulus(stimuli[i].squeeze(), ax=axs[i, 0])
        axs[i, 0].set_title('stimulus')

        # plot example activations
        for k, (channel, ax) in enumerate(zip(channels, axs[i][1:])):
            im = ax.imshow(act[i, channel], vmin=-3, vmax=3, cmap='bwr')
            ax.set_xlabel('x-pos')
            ax.set_ylabel('y-pos')
            ax.set_title(f'channel {channel}')

        cb_ax = fig.add_axes([1.05, 0.8, 0.01, 0.1])
        plt.colorbar(im, cax=cb_ax)
        cb_ax.set_title('activation\n strength')
    plt.show()


# @title Helper Functions

def load_data_split(data_name):
    """Load mouse V1 data from Stringer et al. (2019)

  Data from study reported in this preprint:
  https://www.biorxiv.org/content/10.1101/679324v2.abstract

  These data comprise time-averaged responses of ~20,000 neurons
  to ~4,000 stimulus gratings of different orientations, recorded
  through Calcium imaginge. The responses have been normalized by
  spontaneous levels of activity and then z-scored over stimuli, so
  expect negative numbers. The repsonses were split into train and
  test and then each set were averaged in bins of 6 degrees.

  This function returns the relevant data (neural responses and
  stimulus orientations) in a torch.Tensor of data type torch.float32
  in order to match the default data type for nn.Parameters in
  Google Colab.

  It will hold out some of the trials when averaging to allow us to have test
  tuning curves.

  Args:
    data_name (str): filename to load

  Returns:
    resp_train (torch.Tensor): n_stimuli x n_neurons matrix of neural responses,
        each row contains the responses of each neuron to a given stimulus.
        As mentioned above, neural "response" is actually an average over
        responses to stimuli with similar angles falling within specified bins.
    resp_test (torch.Tensor): n_stimuli x n_neurons matrix of neural responses,
        each row contains the responses of each neuron to a given stimulus.
        As mentioned above, neural "response" is actually an average over
        responses to stimuli with similar angles falling within specified bins
    stimuli: (torch.Tensor): n_stimuli x 1 column vector with orientation
        of each stimulus, in degrees. This is actually the mean orientation
        of all stimuli in each bin.

  """
    with np.load(data_name) as dobj:
        data = dict(**dobj)
    resp_train = data['resp_train']
    resp_test = data['resp_test']
    stimuli = data['stimuli']

    # Return as torch.Tensor
    resp_train_tensor = torch.tensor(resp_train, dtype=torch.float32)
    resp_test_tensor = torch.tensor(resp_test, dtype=torch.float32)
    stimuli_tensor = torch.tensor(stimuli, dtype=torch.float32)

    return resp_train_tensor, resp_test_tensor, stimuli_tensor


def filters(out_channels=6, K=7):
    """ make example filters, some center-surround and gabors
  Returns:
      filters: out_channels x K x K
  """
    grid = np.linspace(-K / 2, K / 2, K).astype(np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing='ij')

    # create center-surround filters
    sigma = 1.1
    gaussian = np.exp(-(xx ** 2 + yy ** 2) ** 0.5 / (2 * sigma ** 2))
    wide_gaussian = np.exp(-(xx ** 2 + yy ** 2) ** 0.5 / (2 * (sigma * 2) ** 2))
    center_surround = gaussian - 0.5 * wide_gaussian

    # create gabor filters
    thetas = np.linspace(0, 180, out_channels - 2 + 1)[:-1] * np.pi / 180
    gabors = np.zeros((len(thetas), K, K), np.float32)
    lam = 10
    phi = np.pi / 2
    gaussian = np.exp(-(xx ** 2 + yy ** 2) ** 0.5 / (2 * (sigma * 0.4) ** 2))
    for i, theta in enumerate(thetas):
        x = xx * np.cos(theta) + yy * np.sin(theta)
        gabors[i] = gaussian * np.cos(2 * np.pi * x / lam + phi)

    filters = np.concatenate((center_surround[np.newaxis, :, :],
                              -1 * center_surround[np.newaxis, :, :],
                              gabors),
                             axis=0)
    filters /= np.abs(filters).max(axis=(1, 2))[:, np.newaxis, np.newaxis]
    filters -= filters.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
    # convert to torch
    filters = torch.from_numpy(filters)
    # add channel axis
    filters = filters.unsqueeze(1)

    return filters


def grating(angle, sf=1 / 28, res=0.1, patch=False):
    """Generate oriented grating stimulus

  Args:
    angle (float): orientation of grating (angle from vertical), in degrees
    sf (float): controls spatial frequency of the grating
    res (float): resolution of image. Smaller values will make the image
      smaller in terms of pixels. res=1.0 corresponds to 640 x 480 pixels.
    patch (boolean): set to True to make the grating a localized
      patch on the left side of the image. If False, then the
      grating occupies the full image.

  Returns:
    torch.Tensor: (res * 480) x (res * 640) pixel oriented grating image

  """

    angle = np.deg2rad(angle)  # transform to radians

    wpix, hpix = 640, 480  # width and height of image in pixels for res=1.0

    xx, yy = np.meshgrid(sf * np.arange(0, wpix * res) / res, sf * np.arange(0, hpix * res) / res)

    if patch:
        gratings = np.cos(
            xx * np.cos(angle + .1) + yy * np.sin(angle + .1))  # phase shift to make it better fit within patch
        gratings[gratings < 0] = 0
        gratings[gratings > 0] = 1
        xcent = gratings.shape[1] * .75
        ycent = gratings.shape[0] / 2
        xxc, yyc = np.meshgrid(np.arange(0, gratings.shape[1]), np.arange(0, gratings.shape[0]))
        icirc = ((xxc - xcent) ** 2 + (yyc - ycent) ** 2) ** 0.5 < wpix / 3 / 2 * res
        gratings[~icirc] = 0.5

    else:
        gratings = np.cos(xx * np.cos(angle) + yy * np.sin(angle))
        gratings[gratings < 0] = 0
        gratings[gratings > 0] = 1

    gratings -= 0.5

    # Return torch tensor
    return torch.tensor(gratings, dtype=torch.float32)


# @title Plotting Functions

def show_stimulus(img, ax=None, show=False):
    """Visualize a stimulus"""
    if ax is None:
        ax = plt.gca()
    ax.imshow(img, cmap=mpl.cm.binary)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if show:
        plt.show()


def plot_corr_matrix(rdm, ax=None, show=False):
    """Plot dissimilarity matrix

  Args:
    rdm (numpy array): n_stimuli x n_stimuli representational dissimilarity
      matrix
    ax (matplotlib axes): axes onto which to plot

  Returns:
    nothing

  """
    if ax is None:
        ax = plt.gca()
    image = ax.imshow(rdm, vmin=0.0, vmax=2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(image, ax=ax, label='dissimilarity')
    if show:
        plt.show()


def plot_multiple_rdm(rdm_dict, resp_dict):
    """Draw multiple subplots for each RDM in rdm_dict."""
    fig, axs = plt.subplots(1, len(rdm_dict),
                            figsize=(4 * len(resp_dict), 3.5))

    # Compute RDM's for each set of responses and plot
    for i, (label, rdm) in enumerate(rdm_dict.items()):
        image = plot_corr_matrix(rdm, axs[i])
        axs[i].set_title(label)
    plt.show()


def plot_rdm_rdm_correlations(rdm_sim):
    """Draw a bar plot showing between-RDM correlations."""
    f, ax = plt.subplots()
    ax.bar(rdm_sim.keys(), rdm_sim.values())
    ax.set_xlabel('Deep network model layer')
    ax.set_ylabel('Correlation of model layer RDM\nwith mouse V1 RDM')
    plt.show()


def plot_rdm_rows(ori_list, rdm_dict, rdm_oris):
    """Plot the dissimilarity of response to each stimulus with response to one
  specific stimulus

  Args:
    ori_list (list of float): plot dissimilarity with response to stimulus with
      orientations closest to each value in this list
    rdm_dict (dict): RDM's from which to extract dissimilarities
    rdm_oris (np.ndarray): orientations corresponding to each row/column of RDMs
    in rdm_dict

  """
    n_col = len(ori_list)
    f, axs = plt.subplots(1, n_col, figsize=(4 * n_col, 4), sharey=True)

    # Get index of orientation closest to ori_plot
    for ax, ori_plot in zip(axs, ori_list):
        iori = np.argmin(np.abs(rdm_oris - ori_plot))

        # Plot dissimilarity curves in each RDM
        for label, rdm in rdm_dict.items():
            ax.plot(rdm_oris, rdm[iori, :], label=label)

        # Draw vertical line at stimulus we are plotting dissimilarity w.r.t.
        ax.axvline(rdm_oris[iori], color=".7", zorder=-1)

        # Label axes
        ax.set_title(f'Dissimilarity with response\nto {ori_plot: .0f}$^o$ stimulus')
        ax.set_xlabel('Stimulus orientation ($^o$)')

    axs[0].set_ylabel('Dissimilarity')
    axs[-1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


class CNN(nn.Module):
    """Deep convolutional network with one convolutional + pooling layer followed
  by one fully connected layer

  Args:
    h_in (int): height of input image, in pixels (i.e. number of rows)
    w_in (int): width of input image, in pixels (i.e. number of columns)

  Attributes:
    conv (nn.Conv2d): filter weights of convolutional layer
    pool (nn.MaxPool2d): max pooling layer
    dims (tuple of ints): dimensions of output from pool layer
    fc (nn.Linear): weights and biases of fully connected layer
    out (nn.Linear): weights and biases of output layer

  """

    def __init__(self, h_in, w_in):
        super().__init__()
        C_in = 1  # input stimuli have only 1 input channel
        C_out = 6  # number of output channels (i.e. of convolutional kernels to convolve the input with)
        K = 7  # size of each convolutional kernel
        Kpool = 8  # size of patches over which to pool
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=K,
                              padding=K // 2)  # add padding to ensure that each channel has same dimensionality as input
        self.pool = nn.MaxPool2d(Kpool)
        self.dims = (C_out, h_in // Kpool, w_in // Kpool)  # dimensions of pool layer output
        self.fc = nn.Linear(np.prod(self.dims), 10)  # flattened pool output --> 10D representation
        self.out = nn.Linear(10, 1)  # 10D representation --> scalar
        self.conv.weight = nn.Parameter(filters(C_out, K))
        self.conv.bias = nn.Parameter(torch.zeros((C_out,), dtype=torch.float32))

    def forward(self, x):
        """Classify grating stimulus as tilted right or left

    Args:
      x (torch.Tensor): p x 48 x 64 tensor with pixel grayscale values for
          each of p stimulus images.

    Returns:
      torch.Tensor: p x 1 tensor with network outputs for each input provided
          in x. Each output should be interpreted as the probability of the
          corresponding stimulus being tilted right.

    """
        x = x.unsqueeze(1)  # p x 1 x 48 x 64, add a singleton dimension for the single stimulus channel
        x = torch.relu(self.conv(x))  # output of convolutional layer
        x = self.pool(x)  # output of pooling layer
        x = x.view(-1, np.prod(self.dims))  # flatten pooling layer outputs into a vector
        x = torch.relu(self.fc(x))  # output of fully connected layer
        x = torch.sigmoid(self.out(x))  # network output
        return x


def train(net, train_data, train_labels,
          n_epochs=25, learning_rate=0.0005,
          batch_size=100, momentum=.99):
    """Run stochastic gradient descent on binary cross-entropy loss for a given
  deep network (cf. appendix for details)

  Args:
    net (nn.Module): deep network whose parameters to optimize with SGD
    train_data (torch.Tensor): n_train x h x w tensor with stimulus gratings
    train_labels (torch.Tensor): n_train x 1 tensor with true tilt of each
      stimulus grating in train_data, i.e. 1. for right, 0. for left
    n_epochs (int): number of times to run SGD through whole training data set
    batch_size (int): number of training data samples in each mini-batch
    learning_rate (float): learning rate to use for SGD updates
    momentum (float): momentum parameter for SGD updates

  """

    # Initialize binary cross-entropy loss function
    loss_fn = nn.BCELoss()

    # Initialize SGD optimizer with momentum
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Placeholder to save loss at each iteration
    track_loss = []

    # Loop over epochs
    for i in range(n_epochs):

        # Split up training data into random non-overlapping mini-batches
        ishuffle = torch.randperm(train_data.shape[0])  # random ordering of training data
        minibatch_data = torch.split(train_data[ishuffle], batch_size)  # split train_data into minibatches
        minibatch_labels = torch.split(train_labels[ishuffle], batch_size)  # split train_labels into minibatches

        # Loop over mini-batches
        for stimuli, tilt in zip(minibatch_data, minibatch_labels):
            # Evaluate loss and update network weights
            out = net(stimuli)  # predicted probability of tilt right
            loss = loss_fn(out, tilt)  # evaluate loss
            optimizer.zero_grad()  # clear gradients
            loss.backward()  # compute gradients
            optimizer.step()  # update weights

            # Keep track of loss at each iteration
            track_loss.append(loss.item())

        # Track progress
        if (i + 1) % (n_epochs // 5) == 0:
            print(f'epoch {i + 1} | loss on last mini-batch: {loss.item(): .2e}')

    print('training done!')


def get_hidden_activity(net, stimuli, layer_labels):
    """Retrieve internal representations of network

  Args:
    net (nn.Module): deep network
    stimuli (torch.Tensor): p x 48 x 64 tensor with stimuli for which to
      compute and retrieve internal representations
    layer_labels (list): list of strings with labels of each layer for which
      to return its internal representations

  Returns:
    dict: internal representations at each layer of the network, in
      numpy arrays. The keys of this dict are the strings in layer_labels.

  """

    # Placeholder
    hidden_activity = {}

    # Attach 'hooks' to each layer of the network to store hidden
    # representations in hidden_activity
    def hook(module, input, output):
        module_label = list(net._modules.keys())[np.argwhere([module == m for m in net._modules.values()])[0, 0]]
        if module_label in layer_labels:  # ignore output layer
            hidden_activity[module_label] = output.view(stimuli.shape[0], -1).detach().numpy()

    hooks = [layer.register_forward_hook(hook) for layer in net.children()]

    # Run stimuli through the network
    pred = net(stimuli)

    # Remove the hooks
    [h.remove() for h in hooks]

    return hidden_activity


if __name__ == '__main__':
    pass
