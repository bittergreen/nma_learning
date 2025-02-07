# @title Import dependencies

# Standard Libraries for file and operating system operations, security, and web requests
import os
import hashlib
import requests
import random
import gc

# Core Python data science and visualization libraries
import numpy as np
import scipy
from matplotlib import pyplot as plt
import logging
from IPython.display import IFrame, display, Image

# Deep Learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import profiler

# Additional utilities
from tqdm.autonotebook import tqdm

# @title Plotting functions

xlim = (-1.8, .7)


def plot_inputs_over_time(timesteps, avg_inputs, title='Inputs over Time'):
    """
    Plot the inputs over time.

    Inputs:
    - timesteps (list or array-like): A sequence of time steps at which the inputs were recorded.
      This acts as the x-axis in the plot, representing the progression of time.
    - avg_inputs (list or array-like): The average values of inputs corresponding to each time step.
      These values are plotted on the y-axis, showing the magnitude of inputs over time.
    - title (string): The title of the plot

    Returns:
    This function generates and displays a plot using Matplotlib.
    """

    with plt.xkcd():
        plt.figure(figsize=(8, 3))
        num_features = avg_inputs.shape[1] if hasattr(avg_inputs, 'shape') else len(avg_inputs[0])

        for feature_idx in range(num_features):
            current_feature_values = avg_inputs[:, feature_idx] if hasattr(avg_inputs, 'shape') else [row[feature_idx]
                                                                                                      for row in
                                                                                                      avg_inputs]
            label = f'Feature {feature_idx + 1}' if feature_idx < num_features - 1 else 'Go Cue'
            plt.plot(timesteps, current_feature_values, label=label)

        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Value (A.U.)')
        plt.subplots_adjust(right=0.7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.tight_layout()
        plt.xlim(min(timesteps), max(timesteps))
        plt.show()


def plot_muscles_over_time(timesteps, avg_output, title='Muscles over Time'):
    """
    Plot the average outputs over time for two muscles to visualize changes in output values.
    The avg_output is expected to be a 250x2 array where each column corresponds to a different muscle.

    Inputs:
    - timesteps (list or array-like): A sequence of time steps at which the outputs were recorded.
      This acts as the x-axis in the plot, representing the progression of time.
    - avg_output (array-like, shape [250, 2]): The average values of outputs, with each column
      representing the output over time for each muscle.
    - title (string): The title of the plot

    Returns:
    This function generates and displays a plot using Matplotlib.
    """

    with plt.xkcd():
        plt.figure(figsize=(8, 3))  # Set the figure size
        plt.plot(timesteps, avg_output[:, 0], label='Muscle 1')  # Plot for muscle 1
        plt.plot(timesteps, avg_output[:, 1], label='Muscle 2')  # Plot for muscle 2
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Value (A.U.)')

        # Adjust plot margins to provide space for the legend outside the plot
        plt.subplots_adjust(right=0.7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)  # Placing legend outside

        plt.tight_layout()
        plt.xlim(min(timesteps), max(timesteps))  # Ensuring x-axis covers the range of timesteps
        plt.show()


def plot_training_validation_losses(epoch_losses, val_losses, actual_num_epochs, title):
    """
    This function plots the training and validation losses over epochs.

    Inputs:
    - epoch_losses (list of float): List containing the training loss for each epoch. Each element is a float
      representing the loss calculated after each epoch of training.
    - val_losses (list of float): List containing the validation loss for each epoch. Similar to `epoch_losses`, but
      for the validation set, allowing for the comparison between training and validation performance.
    - actual_num_epochs (int): The actual number of epochs the training went through. This could be different from
      the initially set number of epochs if early stopping was employed. It determines the range of the x-axis
      in the plot.
    - title (str): A string that sets the title of the plot. This allows for customization of the plot for better
      readability and interpretation.

    Outputs:
    This function generates and displays a plot using matplotlib.
    """

    with plt.xkcd():
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, actual_num_epochs + 1), epoch_losses, label='Training Loss')
        plt.plot(range(1, actual_num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.xlim(xlim)
        plt.show()


# Plot hidden units in UnregularizedRNN
def plot_hidden_unit_activations(hidden_states, times, neurons_to_plot=5, title='PSTHs of Hidden Units'):
    """
    This function plots the average activation of a specified number of neurons from the hidden layers
    of a neural network over a certain number of timesteps.

    Inputs:
        hidden_states (tensor): A 2D tensor containing the hidden states of a network. The dimensions
                                should be (time, features), where 'time' represents the sequence of
                                timesteps, 'batch' represents different data samples, and 'features' represents
                                the neuron activations or features at each timestep.
        times (tensor): The time range that we focus on.
        neurons_to_plot (int, optional): The number of neuron activations to plot, starting from the first neuron.
                                         Defaults to 5.
        title (str, optional): The title of the plot, allowing customization for specific analyses or presentations.
                               Defaults to 'PSTHs of Hidden Units'.

    This function generates and displays a plot of the average activation of specified
    neurons over the selected timesteps, providing a visual analysis of neuron behavior within the network.
    """
    # Apply the nonlinearity to each hidden state before averaging
    rectified_tanh = lambda x: np.where(x > 0, np.tanh(x), 0)
    hidden_states_rectified = rectified_tanh(np.array(hidden_states))

    # Plotting

    with plt.xkcd():
        plt.figure(figsize=(8, 4))
        for i in range(min(neurons_to_plot, hidden_states_rectified.shape[1])):
            plt.plot(times, hidden_states_rectified[:, i], label=f'Neuron {i + 1}')

        plt.xlabel('Time Steps')
        plt.ylabel('Activation')
        plt.title(title)

        # Adjust plot margins to provide space for the legend outside the plot
        plt.subplots_adjust(right=0.8)  # Adjust this value to create more or less space on the right
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')  # Placing legend outside

        plt.xlim(times[0], times[-1])  # Setting x-axis limits based on the provided time tensor
        plt.show()


def plot_psth(data, condition=0, neurons_to_plot=5, title='PSTHs of real data'):
    """
    This function plots PSTHs from real neural data

    Args:
        data (dict): The data from the mat file from Sussillo et al. (2015)
        condition (int, optional): The condition (from 0 to 26). Defaults to 0.
        neurons_to_plot (int, optional): The number of neuron activations to plot, starting from the first neuron.
                                         Defaults to 5.
        title (str, optional): The title for the PSTH plot. This allows users to specify the context or the
                     experiment from which the data is derived.

    Outputs:
    This function directly generates and displays a plot using matplotlib
    to visually represent the neural activity across time bins.
    """
    # Plot
    with plt.xkcd():
        plt.figure(figsize=(8, 4))
        for neuron_idx in range(neurons_to_plot):  # Iterate over each feature/channel
            times_real = data['comboNjs'][0, neuron_idx]['interpTimes'][0]['times'][0].squeeze().astype(float)
            t0 = float(data['comboNjs'][0, neuron_idx]['interpTimes'][0]['moveStarts'][0].item())
            times_real = (times_real - t0) / 1000.0

            spikes_real = data['comboNjs'][0, neuron_idx]['cond'][0]['interpPSTH'][0].squeeze()
            plt.plot(times_real, spikes_real, label=f'Neuron {neuron_idx + 1}')

        plt.xlabel('Time (s)')
        plt.ylabel('Average Activity (Hz)')
        plt.title(title)

        # Adjust plot margins and place legend outside the plot
        plt.subplots_adjust(right=0.8)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

        plt.xlim(times_real[0], times_real[-1])  # Assume times_real is defined
        plt.show()


def plot_perturbation_results(perturbation_strengths, results_regularized, results_unregularized, title):
    """
    This function plots the normalized error percentages of two models (regularized and unregularized) under various
    perturbation strengths.

    Inputs:
        perturbation_strengths (list of float): A list of perturbation strengths tested, representing the
                                                 magnitude of perturbations applied to the model input or parameters.
        results_regularized (list of tuples): Each tuple contains (mean error, standard deviation) for the regularized model
                                         at each perturbation strength.
        results_unregularized (list of tuples): Each tuple contains (mean error, standard deviation) for the unregularized model
                                          at each perturbation strength.
        title (str): The title of the plot, allowing for customization to reflect the analysis context.

    The function generates and displays a bar plot comparing the normalized error
    rates of regularized and unregularized models under different perturbation strengths, with error bars representing the
    standard deviation of errors, normalized to percentage scale.
    """
    mean_errors_regularized, std_errors_regularized = zip(*results_regularized)
    mean_errors_unregularized, std_errors_unregularized = zip(*results_unregularized)

    print("mean_errors_regularized", mean_errors_regularized)
    print("mean_errors_unregularized", mean_errors_unregularized)

    # Plotting

    with plt.xkcd():
        plt.figure(figsize=(8, 6))
        bar_width = 0.35
        bar_positions = np.arange(len(perturbation_strengths))

        plt.bar(bar_positions - bar_width / 2, mean_errors_regularized, width=bar_width, color='blue',
                yerr=std_errors_regularized, capsize=5, label='Regularized Model')
        plt.bar(bar_positions + bar_width / 2, mean_errors_unregularized, width=bar_width, color='red',
                yerr=std_errors_unregularized, capsize=5, label='Unregularized Model')

        plt.xlabel('Perturbation Magnitude')
        plt.ylabel('Normalized Error (%)')
        plt.title(title)
        plt.xticks(bar_positions, [f"{x:.5f}" if x < 0.1 else f"{x}" for x in perturbation_strengths])
        plt.legend()
        plt.ylim(0, 100)
        plt.show()


# @title Set device (GPU or CPU). Execute `set_device()`
# especially if torch modules used.
# @markdown

# inform the user if the notebook uses GPU or CPU.

def set_device():
    """
    Determines and sets the computational device for PyTorch operations based on the availability of a CUDA-capable GPU.

    Outputs:
    - device (str): The device that PyTorch will use for computations ('cuda' or 'cpu'). This string can be directly used
    in PyTorch operations to specify the device.
    """

    if torch.cuda.is_available():
        d = "cuda"
    elif torch.backends.mps.is_available():
        d = "mps"
    else:
        d = "cpu"

    return d


# @title Set random seed, when using `pytorch`

# @markdown Executing `set_seed(seed=seed)` you are setting the seed

# Call `set_seed` function in the exercises to ensure reproducibility.

def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def retrieve_file(fname, url, expected_md5):
    # Check if the file already exists
    if not os.path.isfile(fname):
        try:
            # Attempt to download the file
            response = requests.get(url)
        except requests.ConnectionError:
            # Handle connection errors during the download
            print("!!! Failed to download data !!!")
        else:
            # No connection errors, proceed to check the response
            if response.status_code != requests.codes.ok:
                # Check if the HTTP response status code indicates a successful download
                print("!!! Failed to download data !!!")
            elif hashlib.md5(response.content).hexdigest() != expected_md5:
                # Verify the integrity of the downloaded file using MD5 checksum
                print("!!! Data download appears corrupted !!!")
            else:
                # If download is successful and data is not corrupted, save the file
                with open(fname, "wb") as fid:
                    fid.write(response.content)  # Write the downloaded content to a file


# @title Helper functions

# Define a custom Rectified Tanh activation function
def rectified_tanh(x):
    return torch.where(x > 0, torch.tanh(x), 0)


def grad_rectified_tanh(x):
    return torch.where(x > 0, 1 - torch.tanh(x) ** 2, 0)


def grad_tanh(x):
    return 1 - torch.tanh(x) ** 2


def compute_l2_regularization(parameters, alpha):
    l2_reg = sum(p.pow(2.0).sum() for p in parameters)
    return alpha * l2_reg


def prepare_dataset(file_path, feature_idx=7, muscle_idx=1):
    """
    Load and preprocess data from a .mat file for RNN training.

    Args:
    - file_path: str, path to the .mat file containing the dataset.
    - feature_idx: int, index for individual features for plotting. Max 14.
    - muscle_idx: int, index for muscles for plotting. Max 1.

    Returns:
    - normalised_inputs: Tensor, normalized and concatenated Plan and Go Envelope tensors.
    - avg_output: Tensor, average muscle activity across conditions and delays.
    - timesteps: np.ndarray, array of time steps for plotting.
    """
    # Load the .mat file
    data = scipy.io.loadmat(file_path)

    # Extract condsForSim struct
    conds_for_sim = data['condsForSim']

    # Initialize lists to store data for all conditions
    go_envelope_all, plan_all, muscle_all = [], [], []

    # Get the number of conditions (rows) and delay durations (columns)
    num_conditions, num_delays = conds_for_sim.shape

    times = conds_for_sim['timesREmove'][0][0] / 1000.0

    # Select the same time period as the PSTHs
    rg = slice(46, 296)

    for i in range(num_conditions):  # Loop through each condition
        go_envelope_condition, plan_condition, muscle_condition = [], [], []

        for j in range(num_delays):  # Loop through each delay duration
            condition = conds_for_sim[i, j]
            go_envelope, plan, muscle = condition['goEnvelope'], condition['plan'], condition['muscle']
            selected_muscle_data = muscle[:, [3, 4]]  # Select only specific muscles
            go_envelope_condition.append(go_envelope[rg, :])
            plan_condition.append(plan[rg, :])
            muscle_condition.append(selected_muscle_data[rg, :])

        # Convert lists of arrays to tensors and append to all conditions
        go_envelope_all.append(torch.tensor(np.array(go_envelope_condition), dtype=torch.float32))
        plan_all.append(torch.tensor(np.array(plan_condition), dtype=torch.float32))
        muscle_all.append(torch.tensor(np.array(muscle_condition), dtype=torch.float32))

    times = times[rg]

    # Stack tensors for all conditions
    go_envelope_tensor, plan_tensor, output = torch.stack(go_envelope_all), torch.stack(plan_all), torch.stack(
        muscle_all)

    # Cleanup to free memory
    del data, conds_for_sim, go_envelope_all, plan_all, muscle_all
    gc.collect()

    # Normalize and Standardize Plan Tensor
    plan_tensor = normalize_and_standardize(plan_tensor)

    # Normalise and concatenate Plan and Go Envelope Tensors
    normalised_inputs = normalize_and_standardize(torch.cat([plan_tensor, go_envelope_tensor], dim=3))

    fixed_delay = 3
    inputs_no_delay = normalised_inputs[:, fixed_delay, ...]
    output_no_delay = output[:, fixed_delay, ...]
    return inputs_no_delay, normalised_inputs, output_no_delay, output, times


def normalize_and_standardize(tensor):
    """
    Normalize and standardize a given tensor.

    Args:
    - tensor: Tensor, the tensor to be normalized and standardized.

    Returns:
    - standardized_normalized_tensor: Tensor, the normalized and standardized tensor.
    """
    min_val, max_val = tensor.min(), tensor.max()
    tensor = (tensor - min_val) / (max_val - min_val)  # Normalize
    mean, std = tensor.mean(), tensor.std()
    return (tensor - mean) / std  # Standardize


def train_val_split():
    """Split the data into train and validation splits.
    """
    train_split, val_split = random_split(range(27), [20, 7])
    return sorted(list(train_split)), sorted(list(val_split))


# Can't use mps device due to:
# https://github.com/pytorch/pytorch/issues/95883
# device = set_device()
device = "cpu"
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"


if __name__ == '__main__':
    # List of files to be downloaded with their respective URLs and expected MD5 hashes
    files = [
        ("regularized_model_final.pth", "https://osf.io/kc7sb/download", "9435a9c2ea75766144bf840b25bfb97e"),
        ("unregularized_model_final.pth", "https://osf.io/9vsy5/download", "2e3dc9551b677206e2315788df354a91"),
        ("condsForSimJ2moMuscles.mat", "https://osf.io/wak7e/download", "257d16c4d92759d615bf5cac75dd9a1f"),
        ("m1_reaching_data.mat", "https://osf.io/p2x4n/download", "6fc65443b9632db47772dd2efaadeee0")
    ]

    for fname, url, expected_md5 in files:
        retrieve_file(fname, url, expected_md5)