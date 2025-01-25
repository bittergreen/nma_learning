# Imports
import os
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

from compneuro.w1.d5_deeplearning.utils import filters, grating, plot_example_activations


def download_conv_data():
    # @title Data retrieval and loading
    import hashlib
    import requests

    fname = "W3D4_stringer_oribinned6_split.npz"
    url = "https://osf.io/p3aeb/download"
    expected_md5 = "b3f7245c6221234a676b71a1f43c3bb5"

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


class ConvolutionalLayer(nn.Module):
    """Deep network with one convolutional layer
     Attributes: conv (nn.Conv2d): convolutional layer
    """

    def __init__(self, c_in=1, c_out=6, K=7, filters=None):
        """Initialize layer

        Args:
            c_in: number of input stimulus channels
            c_out: number of output convolutional channels
            K: size of each convolutional filter
            filters: (optional) initialize the convolutional weights

        """
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=K,
                              padding=K // 2, stride=1)
        if filters is not None:
            self.conv.weight = nn.Parameter(filters)
            self.conv.bias = nn.Parameter(torch.zeros((c_out,), dtype=torch.float32))

    def forward(self, s):
        """Run stimulus through convolutional layer

        Args:
            s (torch.Tensor): n_stimuli x c_in x h x w tensor with stimuli

        Returns:
            (torch.Tensor): n_stimuli x c_out x h x w tensor with convolutional layer unit activations.

        """
        a = self.conv(s)  # output of convolutional layer

        return a


if __name__ == '__main__':
    # @markdown Execute this cell to create and visualize filters

    example_filters = filters(out_channels=6, K=7)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(example_filters[0, 0], vmin=-1, vmax=1, cmap='bwr')
    plt.title('center-surround filter')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(example_filters[4, 0], vmin=-1, vmax=1, cmap='bwr')
    plt.title('gabor filter')
    plt.axis('off')
    plt.show()

    # Stimulus parameters
    in_channels = 1  # how many input channels in our images
    h = 48  # height of images
    w = 64  # width of images

    # Convolution layer parameters
    K = 7  # filter size
    out_channels = 6  # how many convolutional channels to have in our layer
    example_filters = filters(out_channels, K)  # create filters to use

    convout = np.zeros(0)  # assign convolutional activations to convout

    # Initialize conv layer and add weights from function filters
    # you need to specify :
    # * the number of input channels c_in
    # * the number of output channels c_out
    # * the filter size K
    convLayer = ConvolutionalLayer(1, 6, K, filters=example_filters)

    # Create stimuli (H_in, W_in)
    orientations = [-90, -45, 0, 45, 90]
    stimuli = torch.zeros((len(orientations), in_channels, h, w), dtype=torch.float32)
    for i, ori in enumerate(orientations):
        stimuli[i, 0] = grating(ori)

    convout = convLayer(stimuli)
    convout = convout.detach()  # detach gradients

    plot_example_activations(stimuli, convout, channels=np.arange(0, out_channels))

