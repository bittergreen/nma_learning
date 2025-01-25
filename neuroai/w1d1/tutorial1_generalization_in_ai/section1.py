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

from neuroai.w1d1.tutorial1_generalization_in_ai.utils import load_pretrained_model

device = "mps" if torch.backends.mps.is_available() else "cpu"
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"


# Define the function to recognize text from an image
def recognize_text(processor, model, image):
    """
    This function takes an image as input and uses a pre-trained language model to generate text from the image.

    Inputs:
    - processor: The processor to use
    - model: The model to use
    - image (PIL Image or Tensor): The input image containing text to be recognized.

    Outputs:
    - text (str): The recognized text extracted from the input image.
    """
    print(image)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values.to(device))
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


if __name__ == '__main__':

    model, processor = load_pretrained_model()

    with gr.Blocks() as demo:
        gr.HTML("<h1>Interactive demo: TrOCR</h1>")
        gr.Markdown("Upload a single image or click one of the examples to try this.")

        # Define the examples
        examples = [
            'neuroai_hello_world.png',
            'sample1.png',
            'sample2.png',
            'sample3.png',
        ]

        # Create the image input component
        image_input = gr.Image(type="pil", label="Upload Image")

        # Create the example gallery
        example_gallery = gr.Examples(
            examples,
            image_input,
        )

        # Create the submit button
        with gr.Row():
            submit_button = gr.Button("Recognize Text", scale=1)

            # Create the text output component
            text_output = gr.Textbox(label="Recognized Text", scale=2)

        # Define the event listeners
        submit_button.click(
            fn=functools.partial(recognize_text, processor, model),
            inputs=image_input,
            outputs=text_output
        )

    # Launch the interface
    demo.launch(height=650)



