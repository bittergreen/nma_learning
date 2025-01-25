# @title Plotting functions
# Standard Libraries for file and operating system operations, security, and web requests
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

device = "mps" if torch.backends.mps.is_available() else "cpu"
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"


def display_image(image_path):
    """Display an image from a given file path.

    Inputs:
    - image_path (str): The path to the image file.
    """
    # Open the image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off the axis
    plt.show()


def display_transformed_images(image, transformations):
    """
    Apply a list of transformations to an image and display them.

    Inputs:
    - image (Tensor): The input image as a tensor.
    - transformations (list): A list of torchvision transformations to apply.
    """
    # Convert tensor image to PIL Image for display
    pil_image = transforms.ToPILImage()(image)

    fig, axs = plt.subplots(len(transformations) + 1, 1, figsize=(5, 15))
    axs[0].imshow(pil_image, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')

    for i, transform in enumerate(transformations):
        # Apply transformation if it's not the placeholder
        if transform != "Custom ElasticTransform Placeholder":
            transformed_image = transform(image)
            # Convert transformed tensor image to PIL Image for display
            display_image = transforms.ToPILImage()(transformed_image)
            axs[i + 1].imshow(display_image, cmap='gray')
            axs[i + 1].set_title(transform.__class__.__name__)
            axs[i + 1].axis('off')
        else:
            axs[i + 1].text(0.5, 0.5, 'ElasticTransform Placeholder', ha='center')
            axs[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def display_original_and_transformed_images(original_tensor, transformed_tensor):
    """
    Display the original and transformed images side by side.

    Inputs:
    - original_tensor (Tensor): The original image as a tensor.
    - transformed_tensor (Tensor): The transformed image as a tensor.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display original image
    original_image = original_tensor.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')

    # Display transformed image
    transformed_image = transformed_tensor.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    axs[1].imshow(transformed_image, cmap='gray')
    axs[1].set_title('Transformed')
    axs[1].axis('off')

    plt.show()


def display_generated_images(generator):
    """
    Display images generated from strings.

    Inputs:
    - generator (GeneratorFromStrings): A generator that produces images from strings.
    """
    plt.figure(figsize=(15, 3))
    for i, (text_img, lbl) in enumerate(generator, 1):
        ax = plt.subplot(1, len(generator.strings) * generator.count // len(generator.strings), i)
        plt.imshow(text_img)
        plt.title(f"Example {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Function to generate an image with text
def generate_image(text, font_path, space_width=2, skewing_angle=8):
    """Generate an image with text.

    Args:
        text (str): Text to be rendered in the image.
        font_path (str): Path to the font file.
        space_width (int): Space width between characters.
        skewing_angle (int): Angle to skew the text image.
    """
    image_size = (350, 50)
    background_color = (255, 255, 255)
    speckle_threshold = 0.05
    speckle_color = (200, 200, 200)
    background = np.random.rand(image_size[1], image_size[0], 1) * 64 + 191
    background = np.tile(background, [1, 1, 4])
    background[:, :, -1] = 255
    image = IMG.fromarray(background.astype('uint8'), 'RGBA')
    image2 = IMG.new('RGBA', image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image2)
    font = ImageFont.truetype(font_path, size=36)
    text_size = draw.textlength(text, font=font)
    text_position = ((image_size[0] - text_size) // 2, (image_size[1] - font.size) // 2)
    draw.text(text_position, text, font=font, fill=(0, 0, 0), spacing=space_width)
    image2 = image2.rotate(skewing_angle)
    image.paste(image2, mask=image2)
    return image


# Function to generate images for multiple strings
def image_generator(strings, font_path, space_width=2, skewing_angle=8):
    """Generate images for multiple strings.

    Args:
        strings (list): List of strings to generate images for.
        font_path (str): Path to the font file.
        space_width (int): Space width between characters.
        skewing_angle (int): Angle to skew the text image.
    """
    for text in strings:
        yield generate_image(text, font_path, space_width, skewing_angle)


# @title Data retrieval

def download_file(fname, url, expected_md5):
    """
    Downloads a file from the given URL and saves it locally.
    Verifies the integrity of the file using an MD5 checksum.

    Args:
    - fname (str): The local filename/path to save the downloaded file.
    - url (str): The URL from which to download the file.
    - expected_md5 (str): The expected MD5 checksum to verify the integrity of the downloaded data.
    """
    if not os.path.isfile(fname):
        try:
            r = requests.get(url)
            r.raise_for_status()  # Raises an HTTPError for bad responses
        except (requests.ConnectionError, requests.HTTPError) as e:
            print(f"!!! Failed to download {fname} due to: {str(e)} !!!")
            return
        if hashlib.md5(r.content).hexdigest() == expected_md5:
            with open(fname, "wb") as fid:
                fid.write(r.content)
            print(f"{fname} has been downloaded successfully.")
        else:
            print(f"!!! Data download appears corrupted, {hashlib.md5(r.content).hexdigest()} !!!")


def extract_zip(zip_fname, folder='.'):
    """
    Extracts a ZIP file to the specified folder.

    Args:
    - zip_fname (str): The filename/path of the ZIP file to be extracted.
    - folder (str): Destination folder where the ZIP contents will be extracted.
    """
    if zipfile.is_zipfile(zip_fname):
        with zipfile.ZipFile(zip_fname, 'r') as zip_ref:
            zip_ref.extractall(folder)
            print(f"Extracted {zip_fname} to {folder}.")
    else:
        print(f"Skipped extraction for {zip_fname} as it is not a zip file.")


def load_pretrained_model():
    # Load the pre-trained TrOCR model and processor
    # with contextlib.redirect_stdout(io.StringIO()):
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.to(device=device)
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten", use_fast=False)
    return model, processor


def get_images_and_transcripts(df, subject):
    df_ = df[df.subject == subject]
    transcripts = df_.transcript.values.tolist()

    # Load the corresponding images
    images = []
    for _, row in df_.iterrows():
        images.append(IMG.open(row.filename))

    return images, transcripts


if __name__ == '__main__':
    # Define the list of files to download, including both ZIP files and other file types
    file_info = [
        ("Dancing_Script.zip", "https://osf.io/32yed/download", "d59bd3201b58a37d0d3b4cd0b0ec7400", '.'),
        ("lines.zip", "https://osf.io/8a753/download", "6815ed3987f8eb2fd3bc7678c11f2e9e", 'lines'),
        ("transcripts.csv", "https://osf.io/9hgr8/download", "d81d9ade10db55603cc893345debfaa2", None),
        ("neuroai_hello_world.png", "https://osf.io/zg4w5/download", "f08b81e47f2fe66b5f25b2ccc204c780", None),
        # New image file
        ("sample0.png",
         "https://github.com/neuromatch/NeuroAI_Course/blob/main/tutorials/W1D1_Generalization/static/sample_0.png?raw=true",
         '920ae567f707bfee0be29dc854f804ed', None),
        ("sample1.png",
         "https://github.com/neuromatch/NeuroAI_Course/blob/main/tutorials/W1D1_Generalization/static/sample_1.png?raw=true",
         'cd28623a829b40d0a1dd8c0f17e9ebd7', None),
        ("sample2.png",
         "https://github.com/neuromatch/NeuroAI_Course/blob/main/tutorials/W1D1_Generalization/static/sample_2.png?raw=true",
         'c189c09abf989eac4e1a8d493bd362d7', None),
        ("sample3.png",
         "https://github.com/neuromatch/NeuroAI_Course/blob/main/tutorials/W1D1_Generalization/static/sample_3.png?raw=true",
         'dcffc678266952f18af1fc1242127e98', None)
    ]

    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        # Process the downloads and extractions
        for fname, url, expected_md5, folder in file_info:
            download_file(fname, url, expected_md5)
            if folder is not None:
                extract_zip(fname, folder)

