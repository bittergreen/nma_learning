# Standard Libraries for file and operating system operations, security, and web requests
import contextlib
import os
import functools
import hashlib
import requests
import logging
import io
import re
import time

# Core python data science and image processing libraries
import numpy as np
from PIL import Image as IMG
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import tqdm

# Deep Learning and model specific libraries
import torch
import torchmetrics.functional.text as fm
import transformers
from torchvision import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Utility and interface libraries
import gradio as gr
from IPython.display import IFrame, display, Image
import sentencepiece
import zipfile
import pandas as pd

from neuroai.w1d1.tutorial1_generalization_in_ai.utils import load_pretrained_model, get_images_and_transcripts, \
    display_transformed_images, image_generator

device = "mps" if torch.backends.mps.is_available() else "cpu"
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"


def transform_images():
    # Convert PIL Image to Tensor
    image = IMG.open("neuroai_hello_world.png")
    image = transforms.ToTensor()(image)

    # Define each transformation separately
    # RandomAffine: applies rotations, translations, scaling. Here, rotates by up to ±15 degrees,
    affine = transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1))

    # ElasticTransform: applies elastic distortions to the image. The 'alpha' parameter controls
    # the intensity of the distortion.
    elastic = transforms.ElasticTransform(alpha=25.0)

    # RandomPerspective: applies random perspective transformations with a specified distortion scale.
    perspective = transforms.RandomPerspective(distortion_scale=0.2, p=1.0)

    # RandomErasing: randomly erases a rectangle area in the image.
    erasing = transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False)

    # GaussianBlur: applies gaussian blur with specified kernel size and sigma range.
    gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.8, 5))

    # A list of all transformations for iteration
    transformations = [affine, elastic, perspective, erasing, gaussian_blur]

    # Display
    display_transformed_images(image, transformations)


def gen_synthetic_data():
    # Define strings
    strings = ['Hello world', 'This is the first tutorial', 'For Neuromatch NeuroAI']

    # Specify font path
    font_path = "DancingScript-VariableFont_wght.ttf"  # Ensure this path is correct

    # Example usage
    strings = ['Hello world', 'This is the first tutorial', 'For Neuromatch NeuroAI']
    font_path = "DancingScript-VariableFont_wght.ttf"  # Ensure this path is correct

    # Create a generator with the specified parameters
    generator = image_generator(strings, font_path, space_width=2, skewing_angle=3)

    i = 1
    for img in generator:
        plt.imshow(img, cmap='gray')
        plt.title(f"Example {i}")
        plt.axis('off')
        plt.show()
        i += 1


if __name__ == '__main__':
    transform_images()
    gen_synthetic_data()

