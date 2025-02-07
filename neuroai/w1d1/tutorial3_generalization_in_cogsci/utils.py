# @title Import dependencies
import contextlib
import io
# Standard libraries
import hashlib
import logging
import os
import random
from tkinter import *

import requests
import shutil
import time
from importlib import reload
import zipfile
from zipfile import ZipFile

# Data handling and visualization
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage
from sklearn.model_selection import train_test_split

# Deep Learning libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

# Interactive controls in Jupyter notebooks
from IPython.display import clear_output, display, update_display
import ipywidgets as widgets

# Utility for progress bars
from tqdm import tqdm


# Current directory
base_dir = os.getcwd()


# @title Plotting functions
# @markdown

def display_images(probe, options):
    # Open the probe image and the option images
    probe_image = Image.open(probe)
    option_images = [Image.open(img_path) for img_path in options]

    # Create a figure with the probe and the 3x3 grid for the options directly below
    fig = plt.figure(figsize=(15, 10))  # Adjust figure size as needed

    # Add the probe image to the top of the figure with a red border
    ax_probe = fig.add_subplot(4, 3, (1, 3))  # Span the probe across the top 3 columns
    ax_probe.imshow(probe_image)
    ax_probe.axis('off')
    rect = patches.Rectangle((0, 0), probe_image.width-1, probe_image.height-1, linewidth=2, edgecolor='r', facecolor='none')
    ax_probe.add_patch(rect)

    # Position the 3x3 grid of option images directly below the probe image
    for index, img in enumerate(option_images):
        row = (index // 3) + 1  # Calculate row in the 3x3 grid, starting directly below the probe
        col = (index % 3) + 1   # Calculate column in the 3x3 grid
        ax_option = fig.add_subplot(4, 3, row * 3 + col)  # Adjust grid position to directly follow the probe
        ax_option.imshow(img)
        ax_option.axis('off')

    plt.tight_layout()
    plt.show()


# @title Data retrieval for zip files

def handle_file_operations(fname, url, expected_md5, extract_to='data'):
    """Handles downloading, verifying, and extracting a file."""

    # Define helper functions for download, verify, and extract operations
    def download_file(url, filename):
        """Downloads file from the given URL and saves it locally."""
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(filename, "wb") as fid:
                for chunk in r.iter_content(chunk_size=8192):
                    fid.write(chunk)
            print("Download successful.")
            return True
        except requests.RequestException as e:
            print(f"!!! Failed to download data: {e} !!!")
            return False

    def verify_file_md5(filename, expected_md5):
        """Verifies the file's MD5 checksum."""
        hash_md5 = hashlib.md5()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        if hash_md5.hexdigest() == expected_md5:
            print("MD5 checksum verified.")
            return True
        else:
            print("!!! Data download appears corrupted !!!")
            return False

    def extract_zip_file(filename, extract_to):
        """Extracts the ZIP file to the specified directory."""
        try:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"File extracted successfully to {extract_to}")
        except zipfile.BadZipFile:
            print("!!! The ZIP file is corrupted or not a zip file !!!")

    # Main operation
    if not os.path.isfile(fname) or not verify_file_md5(fname, expected_md5):
        if download_file(url, fname) and verify_file_md5(fname, expected_md5):
            extract_zip_file(fname, extract_to)
    else:
        print(f"File '{fname}' already exists and is verified. Proceeding to extraction.")
        extract_zip_file(fname, extract_to)


# @title Data retrieval for torch models

def download_file(url, filename):
    """
    Download a file from a given URL and save it in the specified directory.
    """
    filepath = os.path.join(base_dir, filename)  # Ensure the file is saved in base_dir

    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP request errors

    with open(filepath, 'wb') as f:
        f.write(response.content)


def verify_checksum(filename, expected_checksum):
    """
    Verify the MD5 checksum of a file

    Parameters:
    filename (str): Path to the file
    expected_checksum (str): Expected MD5 checksum

    Returns:
    bool: True if the checksum matches, False otherwise
    """
    md5 = hashlib.md5()

    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)

    return md5.hexdigest() == expected_checksum


def load_models(model_files, directory, map_location='cpu'):
    """
    Load multiple models from a specified directory.
    """
    models = {}
    for model_file in model_files:
        full_path = os.path.join(directory, model_file)  # Correctly join paths
        models[model_file] = torch.load(full_path, map_location=map_location)
    return models


def verify_models_in_destination(model_files, destination_directory):
    """
    Verify the presence of model files in the specified directory.

    Parameters:
    model_files (list of str): Filenames of the models to verify.
    destination_directory (str): The directory where the models are supposed to be.

    Returns:
    bool: True if all models are found in the directory, False otherwise.
    """
    missing_files = []
    for model_file in model_files:
        # Construct the full path to where the model should be
        full_path = os.path.join(destination_directory, model_file)
        # Check if the model exists at the location
        if not os.path.exists(full_path):
            missing_files.append(model_file)

    if missing_files:
        print(f"Missing model files in destination: {missing_files}")
        return False
    else:
        print("All models are correctly located in the destination directory.")
        return True


# @title Helper functions

def select_random_images_within_alphabet(base_path, alphabet_path, exclude_character_path, num_images=8):
    # Initialize an empty list to store the paths of the chosen images
    chosen_images = []

    # Get a list of all character directories within the alphabet_path, excluding the directory specified by exclude_character_path
    all_characters = [
        char for char in os.listdir(alphabet_path)
        if os.path.isdir(os.path.join(alphabet_path, char)) and os.path.join(alphabet_path, char) != exclude_character_path
    ]

    # Keep selecting images until we have the desired number of images (num_images)
    while len(chosen_images) < num_images:
        # If there are no more characters to choose from, exit the loop
        if not all_characters:
            break

        # Randomly select a character directory from the list of all characters
        character = random.choice(all_characters)
        # Construct the full path to the selected character directory
        character_path = os.path.join(alphabet_path, character)

        # Get a list of all image files (with .png extension) in the selected character directory
        all_images = [
            img for img in os.listdir(character_path)
            if img.endswith('.png')
        ]

        # If there are no images in the selected character directory, continue to the next iteration
        if not all_images:
            continue

        # Randomly select an image file from the list of image files
        image_file = random.choice(all_images)
        # Construct the full path to the selected image file
        image_path = os.path.join(character_path, image_file)

        # Add the selected image path to the list of chosen images
        chosen_images.append(image_path)

    # Return the list of paths to the chosen images
    return chosen_images


def get_example_data():
    # Example usage
    file_info = [
        {"fname": "omniglot-py.zip", "url": "https://osf.io/bazxp/download", "expected_md5": "f7a4011f5c25460c6d95ee1428e377ed"},
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        for file in file_info:
            handle_file_operations(**file)


def get_model():
    # URLs and checksums for the models
    models_info = {
        'location_model.pt': ('https://osf.io/zmd7y/download', 'dfd51cf7c3a277777ad941c4fcc23813'),
        'stroke_model.pt': ('https://osf.io/m6yc7/download', '511ea7bd12566245d5d11a85d5a0abb0'),
        'terminate_model.pt': ('https://osf.io/dsmhc/download', '2f3e26cfcf36ce9f9172c15d8b1079d1')
    }

    destination_directory = base_dir

    # Define model_files based on the keys of models_info to ensure we have the filenames
    model_files = list(models_info.keys())

    with contextlib.redirect_stdout(io.StringIO()):
        # Iterate over the models to download and verify
        for model_name, (url, checksum) in models_info.items():
            download_file(url, model_name)  # Downloads directly into base_dir
            if verify_checksum(os.path.join(base_dir, model_name), checksum):
                print(f"Successfully verified {model_name}")
            else:
                print(f"Checksum does not match for {model_name}. Download might be corrupted.")

    with contextlib.redirect_stdout(io.StringIO()):
        # Verify the presence of the models in the destination directory
        if verify_models_in_destination(model_files, destination_directory):
            print("Verification successful: All models are in the correct directory.")
        else:
            print("Verification failed: Some models are missing from the destination directory.")

    # Load the models from the destination directory
    models = load_models(model_files, destination_directory, map_location='cpu')


if __name__ == '__main__':
    get_example_data()
    get_model()

