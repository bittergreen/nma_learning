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
from neuroai.w1d1.tutorial2_generalization_in_neuroscience.utils import *


class TimeseriesDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        inputs: Tensor of shape [#examples, time, input_features]
        targets: Tensor of shape [#examples, time, output_features]
        """
        self.inputs = inputs
        self.targets = targets
        self.num_conditions = inputs.shape[0]
        assert inputs.shape[0] == targets.shape[0]

    def __len__(self):
        return self.num_conditions

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        target_seq = self.targets[idx]
        return input_seq, target_seq


class UnregularizedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, g, h, tau_over_dt=5):
        super(UnregularizedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.tau_over_dt = tau_over_dt
        self.output_linear = nn.Linear(hidden_size, output_size)

        # Weight initialization
        self.J = nn.Parameter(torch.randn(hidden_size, hidden_size) * (g / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float))))
        self.B = nn.Parameter(torch.randn(hidden_size, input_size) * (h / torch.sqrt(torch.tensor(input_size, dtype=torch.float))))
        self.bx = nn.Parameter(torch.zeros(hidden_size))

        # Nonlinearity
        self.nonlinearity = rectified_tanh

    def forward(self, input, hidden):

        # Calculate the visible firing rate from the hidden state.
        firing_rate_before = self.nonlinearity(hidden)

        # Update hidden state
        recurrent_drive = torch.matmul(self.J, firing_rate_before.transpose(0, 1))
        input_drive = torch.matmul(self.B, input.transpose(0, 1))
        total_drive = recurrent_drive + input_drive + self.bx.unsqueeze(1)
        total_drive = total_drive.transpose(0, 1)

        # Euler integration for continuous-time update
        hidden = hidden + (1 / self.tau_over_dt) * (-hidden + total_drive)

        # Calculate the new firing rate given the update.
        firing_rate = self.nonlinearity(hidden)

        # Project the firing rate linearly to form the output
        output = self.output_linear(firing_rate)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)


class RegularizedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, g, h, tau_over_dt=5):
        super(RegularizedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.tau_over_dt = tau_over_dt  # Time constant
        self.output_linear = nn.Linear(hidden_size, output_size)

        # Weight initialization
        self.J = nn.Parameter(torch.randn(hidden_size, hidden_size) * (g / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float))))
        self.B = nn.Parameter(torch.randn(hidden_size, input_size) * (h / torch.sqrt(torch.tensor(input_size, dtype=torch.float))))
        self.bx = nn.Parameter(torch.zeros(hidden_size))

        # Nonlinearity
        self.nonlinearity = rectified_tanh

    def forward(self, input, hidden):
        # Calculate the visible firing rate from the hidden state.
        firing_rate_before = self.nonlinearity(hidden)

        # Update hidden state
        recurrent_drive = torch.matmul(self.J, firing_rate_before.transpose(0, 1))
        input_drive = torch.matmul(self.B, input.transpose(0, 1))
        total_drive = recurrent_drive + input_drive + self.bx.unsqueeze(1)
        total_drive = total_drive.transpose(0, 1)

        # Euler integration for continuous-time update
        hidden = hidden + (1 / self.tau_over_dt) * (-hidden + total_drive)

        # Calculate the new firing rate given the update.
        firing_rate = self.nonlinearity(hidden)

        # Project the firing rate linearly to form the output
        output = self.output_linear(firing_rate)

        # Regularization terms (used for R1 calculation)
        firing_rate_reg = firing_rate.pow(2).sum()

        return output, hidden, firing_rate_reg

    def init_hidden(self, batch_size):
        # Initialize hidden state with batch dimension
        return torch.zeros(batch_size, self.hidden_size).to(device)


def generate_trajectory(model, inputs, device):
    inputs = inputs.to(device)
    batch_size = inputs.size(0)
    h = model.init_hidden(batch_size)

    loss = 0
    outputs = []
    hidden_states = []
    with torch.no_grad():
        for t in range(inputs.shape[1]):
            # Forward the model's input and hidden state to obtain the model
            # output and hidden state *h*.
            # Note that you should index the input tensor by the time dimension
            # Capture any additional outputs in 'rest'
            output, h, *rest = model(inputs[:, t], h)
            outputs.append(output)
            hidden_states.append(h.detach().clone())
            print(t)

    return torch.stack(outputs, axis=1).to(device), torch.stack(hidden_states, axis=1).to(device)


def plot_data(normalised_inputs, times, outputs):
    # Averaging across conditions and delays
    reach_directions = [0, 6, 9]
    for reach in reach_directions:
        # Plot inputs and outputs
        one_direction = normalised_inputs[reach, ...].clone()
        # Exaggerate the go cue
        one_direction[:, -1] *= 5
        plot_inputs_over_time(times, one_direction, title=f'Inputs over Time, reach direction {reach}')
        plot_muscles_over_time(times, outputs[reach, ...], title=f'Outputs over Time, reach direction {reach}')


def preparation():
    # Define the path to the dataset file containing conditions for simulation of muscles
    file_path = 'condsForSimJ2moMuscles.mat'

    # Prepare the dataset by loading and processing it from the specified file path
    normalised_inputs, normalised_inputs_with_delay, outputs, outputs_with_delay, times = prepare_dataset(file_path)

    print("Shape of the inputs", normalised_inputs.shape)
    print("Shape of the output", outputs.shape)
    # plot_data(normalised_inputs, times, outputs)

    # Create the dataset with the fixed delay
    train_idx, val_idx = train_val_split()
    train_dataset = TimeseriesDataset(normalised_inputs[train_idx], outputs[train_idx])
    val_dataset = TimeseriesDataset(normalised_inputs[val_idx], outputs[val_idx])

    # Create DataLoaders
    batch_size = 20
    unregularized_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    unregularized_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = 16
    hidden_size = 10
    output_size = 2
    g = 4
    h_val = 1.0

    model = UnregularizedRNN(input_size, hidden_size, output_size, g, h_val)
    model.to(device)

    for inputs, targets in unregularized_train_loader:
        hidden = model.init_hidden(batch_size)
        output, hidden_after = model(inputs[:, 0, :].to(device), hidden.to(device))
        assert output.shape == targets[:, 0].shape
        assert hidden_after.shape == hidden.shape
        break

    trajectory, hidden_states = generate_trajectory(model,
                                                    inputs[0].unsqueeze(0),
                                                    device)

    with plt.xkcd():
        plot_hidden_unit_activations(hidden_states=hidden_states.squeeze().detach().cpu().numpy(),
                                     times=times,
                                     neurons_to_plot=7,
                                     title='Hidden units')
        plot_muscles_over_time(times, trajectory.squeeze().detach().cpu().numpy(), 'Generated muscle activity')


def run_unregularized():
    # Instantiate model
    input_size = 16
    hidden_size = 150
    output_size = 2  # Number of muscles
    g = 4  # g value
    h_val = 1.0  # h value

    unregularized_model = UnregularizedRNN(input_size, hidden_size, output_size, g, h_val)
    unregularized_model.to(device)  # Move model to the appropriate device

    # Load the pretrained model
    model_path = 'unregularized_model_final.pth'
    model_state_dict = torch.load(model_path, map_location=device)
    unregularized_model.load_state_dict(model_state_dict)
    unregularized_model.eval()  # Set model to evaluation mode

    # Example index
    idx = 0

    # load data
    file_path = 'condsForSimJ2moMuscles.mat'
    normalised_inputs, normalised_inputs_with_delay, outputs, outputs_with_delay, times = prepare_dataset(file_path)
    train_idx, val_idx = train_val_split()
    # Ensure data is on the correct device
    sample_input = normalised_inputs[train_idx[idx], ...].to(device)
    sample_target = outputs[train_idx[idx], ...].to(device)

    # Generate trajectory
    generated_target, hidden_states = generate_trajectory(unregularized_model, sample_input.unsqueeze(0), device)

    # Plotting
    plot_inputs_over_time(times, sample_input.cpu())
    plot_muscles_over_time(times, sample_target.cpu(), 'Targets')
    plot_muscles_over_time(times, generated_target.squeeze().detach().cpu().numpy(), 'Generated')


def run_regularized():
    # load data
    file_path = 'condsForSimJ2moMuscles.mat'
    normalised_inputs, normalised_inputs_with_delay, outputs, outputs_with_delay, times = prepare_dataset(file_path)
    train_idx, val_idx = train_val_split()

    train_flattened_inputs = normalised_inputs_with_delay[train_idx].view(-1, *normalised_inputs_with_delay.shape[2:])
    train_flattened_targets = outputs_with_delay[train_idx].view(-1, *outputs_with_delay.shape[2:])

    val_flattened_inputs = normalised_inputs_with_delay[val_idx].view(-1, *normalised_inputs_with_delay.shape[2:])
    val_flattened_targets = outputs_with_delay[val_idx].view(-1, *outputs_with_delay.shape[2:])

    # Create the dataset with the fixed delay
    train_dataset = TimeseriesDataset(train_flattened_inputs, train_flattened_targets)
    val_dataset = TimeseriesDataset(val_flattened_inputs, val_flattened_targets)

    # Create DataLoaders
    batch_size = 20
    regularized_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    regularized_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate model
    input_size = 16  # Features + Go Cue
    hidden_size = 150
    output_size = 2  # Number of muscles
    g = 1.5  # g value
    h_val = 1.0  # h value

    regularized_model = RegularizedRNN(input_size, hidden_size, output_size, g, h_val)
    regularized_model.to(device)  # Move model to the appropriate device

    # Load the pretrained model
    model_path = 'regularized_model_final.pth'
    model_state_dict = torch.load(model_path, map_location=device)
    regularized_model.load_state_dict(model_state_dict)
    regularized_model.eval()  # Set model to evaluation mode

    # Example index
    idx = 0

    # Ensure data is on the correct device
    sample_input = normalised_inputs[train_idx[idx], ...].to(device)
    sample_target = outputs[train_idx[idx], ...].to(device)

    # Generate trajectory
    generated_target, hidden_states = generate_trajectory(regularized_model, sample_input.unsqueeze(0), device)

    # Plotting
    plot_inputs_over_time(times, sample_input.cpu())
    plot_muscles_over_time(times, sample_target.cpu(), 'Targets')
    plot_muscles_over_time(times, generated_target.squeeze().detach().cpu().numpy(), 'Generated')

    plot_hidden_unit_activations(hidden_states=hidden_states.squeeze().detach().cpu().numpy(),
                                 times=times,
                                 neurons_to_plot=10,
                                 title='PSTHs of Hidden Units in RegularizedRNN')

    data = scipy.io.loadmat('m1_reaching_data.mat')
    plot_psth(data, neurons_to_plot=10)


def perturb_inputs(model, inputs, perturbation_strength):
    # Perturb the inputs by adding random noise scaled by the perturbation strength and input strength
    input_strength = torch.norm(inputs, p=2, dim=-1, keepdim=True)  # Calculate the L2 norm of inputs
    noise = torch.rand(inputs.shape[0], 1, inputs.shape[2], device=device) * perturbation_strength * input_strength
    perturbed_inputs = inputs + noise
    return perturbed_inputs


def compute_loss(model, inputs, targets, criterion, device):
    batch_size = inputs.size(0)
    h = model.init_hidden(batch_size).to(device)  # Initialize hidden state
    losses = []
    for t in range(inputs.shape[1]):  # Iterate over time steps
        model_output = model(inputs[:, t, :], h)
        output, h, *rest = model_output[:2]
        loss = criterion(output, targets[:, t])  # Assume targets is a sequence of same length as inputs
        losses.append(loss)
    mean_loss = torch.mean(torch.stack(losses)).item()
    return mean_loss


def test_perturbed_inputs(model, perturbation_strengths, test_loader, criterion, device, max_error):
    model.eval()  # Set the model to evaluation mode
    perturbation_results = []

    for strength in perturbation_strengths:
        all_errors = []  # Store all errors for each perturbation strength to compute mean and s.d.
        print(f"Testing perturbation strength {strength}")
        for iteration in tqdm(range(30)):  # Repeat the procedure 30 times
            batch_errors = []  # Store errors for each batch

            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # Compute error for original inputs
                # original_loss = compute_loss(model, inputs, targets, criterion, device)
                # Compute error for perturbed inputs

                perturbed_inputs = perturb_inputs(model, inputs, strength)
                perturbed_loss = compute_loss(model, perturbed_inputs, targets, criterion, device)

                # Store the normalized error.
                rel_error = perturbed_loss / max_error * 100
                batch_errors.append(rel_error)

            all_errors.extend(batch_errors)

        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        perturbation_results.append((mean_error, std_error))
        print(f"Completed testing for perturbation strength {strength}.")

    return perturbation_results


def calculate_mean_absolute_strength(model):
    # Calculate the mean absolute connection strength of the recurrent weight matrix
    return torch.mean(torch.abs(model.J)).item()


def perturb_recurrent_weights(model, mean_strength, perturbation_percentage):
    perturbation_strength = mean_strength * perturbation_percentage
    with torch.no_grad():
        noise = torch.randn_like(model.J) * perturbation_strength
        perturbed_weights = model.J + noise
        return perturbed_weights


def test_perturbed_structure(model, perturbation_percentages, test_loader, criterion, device, max_error):

    model.eval()  # Set the model to evaluation mode
    mean_strength = calculate_mean_absolute_strength(model)
    perturbation_results = []  # List to store (mean error, std dev) tuples

    original_weights = model.J.data.clone()  # Save the original weights

    for percentage in perturbation_percentages:
        multiple_perturbations_error = []
        print(f"Testing perturbation percentage {percentage:.4f}")

        for perturbation in tqdm(range(30)):  # Perturb 30 times for each strength
            batch_errors = []
            perturbed_weights = perturb_recurrent_weights(model, mean_strength, percentage)
            model.J.data = perturbed_weights.data

            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)
                h = model.init_hidden(batch_size).to(device)

                outputs = torch.zeros_like(targets).to(device)
                for t in range(inputs.shape[1]):
                    output, h, *rest = model(inputs[:, t, :], h)
                    outputs[:, t, :] = output

                loss = criterion(outputs, targets).item()
                batch_errors.append(loss)

            # Reset to original weights after each perturbation
            model.J.data = original_weights.data
            multiple_perturbations_error.append(np.mean(batch_errors))

        mean_error = np.mean(multiple_perturbations_error)  # Average over the 50 perturbations
        std_dev_error = np.std(multiple_perturbations_error)  # Standard deviation for error bars
        perturbation_results.append((100 * mean_error / max_error, 100 * std_dev_error / max_error))

        # Normalize the errors
        print(f"Completed testing for perturbation percentage {percentage:.4f}. Mean error: {mean_error:.4f}, Std. dev.: {std_dev_error:.4f}\n")

    return perturbation_results


def test_robustness():
    # load data
    file_path = 'condsForSimJ2moMuscles.mat'
    normalised_inputs, normalised_inputs_with_delay, outputs, outputs_with_delay, times = prepare_dataset(file_path)
    train_idx, val_idx = train_val_split()

    batch_size = 20

    test_dataset = TimeseriesDataset(normalised_inputs[train_idx], outputs[train_idx])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate model
    input_size = 16
    hidden_size = 150
    output_size = 2  # Number of muscles
    h_val = 1.0  # h value

    unregularized_model = UnregularizedRNN(input_size, hidden_size, output_size, g=4, h=h_val)
    unregularized_model.to(device)  # Move model to the appropriate device
    # Load the pretrained model
    model_path = 'unregularized_model_final.pth'
    model_state_dict = torch.load(model_path, map_location=device)
    unregularized_model.load_state_dict(model_state_dict)
    unregularized_model.eval()  # Set model to evaluation mode

    regularized_model = RegularizedRNN(input_size, hidden_size, output_size, g=1.5, h=h_val)
    regularized_model.to(device)  # Move model to the appropriate device
    # Load the pretrained model
    model_path = 'regularized_model_final.pth'
    model_state_dict = torch.load(model_path, map_location=device)
    regularized_model.load_state_dict(model_state_dict)
    regularized_model.eval()  # Set model to evaluation mode

    # Calculate the maximum error for a null model, the error when the output is constant.
    max_error = ((outputs - outputs.mean(axis=[0, 1], keepdims=True)) ** 2).mean()

    perturbation_strengths = [0.0125, 0.025, 0.05, 0.1, 0.2]
    results_unregularized = test_perturbed_inputs(unregularized_model, perturbation_strengths, test_loader,
                                                  nn.MSELoss(), device, max_error)
    results_regularized = test_perturbed_inputs(regularized_model, perturbation_strengths, test_loader, nn.MSELoss(),
                                                device, max_error)

    # Plot perturbation results
    plot_perturbation_results(perturbation_strengths, results_regularized, results_unregularized,
                              "Perturbation of the inputs")


if __name__ == '__main__':
    # run_regularized()
    test_robustness()

