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

from neuroai.w1d1.tutorial1_generalization_in_ai.utils import load_pretrained_model, get_images_and_transcripts

device = "mps" if torch.backends.mps.is_available() else "cpu"
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7897"


def inspect_encoder(model):
    """
    Inspect encoder to verify that it processes inputs in the expected way.

    Args:
        model: the TrOCR model
    """
    single_input = torch.zeros(1, 3, 384, 384).to(device)

    # Run the input through the encoder.
    output = model.encoder(single_input)

    # Measure the number of hidden tokens which are the output of the encoder
    hidden_shape = output['last_hidden_state'].shape

    assert hidden_shape[0] == 1
    assert hidden_shape[1] == 577
    assert hidden_shape[2] == 768


def more_inspects(images, model, processor):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    pixel_values = processor(images=[images[0]], return_tensors="pt").pixel_values
    encoded_image = model.encoder(pixel_values.to(device))
    print(encoded_image.last_hidden_state.shape)

    decoded = model.decoder.forward(
        input_ids=torch.Tensor([[0]]).to(device, dtype=int),
        encoder_hidden_states=encoded_image['last_hidden_state'],
    )
    print(decoded.logits.shape)
    print(decoded.logits.argmax())
    print(processor.tokenizer.decode(31206))

    decoded = model.decoder.forward(
        input_ids=torch.Tensor([[0, 31206]]).to(device, dtype=int),
        encoder_hidden_states=encoded_image['last_hidden_state'],
    )
    processor.tokenizer.decode(decoded.logits[:, -1, :].argmax().item())

    # Move the model to the appropriate device
    model.to(device)

    # move it to the same device
    pixel_values = pixel_values.to(device)

    # Generate the sequence using the model
    best_sequence = model.generate(pixel_values).to(device)

    # Decode the generated sequence
    decoded_sequence = processor.tokenizer.decode(best_sequence[0])
    print(decoded_sequence)

    return best_sequence, encoded_image


def visualize_attention(decoded, layer=7, head=5):
    plt.figure(figsize=(10, 10))

    image = images[0]
    for token in range(decoded.cross_attentions[layer].shape[2]):
        attention_pattern = decoded.cross_attentions[layer][0, head, token, 1:].reshape((24, 24))
        attention_pattern = attention_pattern.detach().cpu().numpy()

        print(processor.decode(best_sequence[0][:token+1]))
        plt.imshow((np.array(image).mean(axis=2)).astype(float), cmap='gray')
        plt.imshow(attention_pattern, extent=[0, image.width, 0, image.height], alpha=attention_pattern/attention_pattern.max(), cmap='YlOrRd')
        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.show()


def inspect_attention():
    decoded = model.decoder.forward(
        input_ids=best_sequence,
        encoder_hidden_states=encoded_image['last_hidden_state'],
        output_attentions=True
    )
    visualize_attention(decoded)


if __name__ == '__main__':
    model, processor = load_pretrained_model()
    print(model.encoder)
    print(model.decoder)
    inspect_encoder(model)

    df = pd.read_csv('transcripts.csv')
    df['filename'] = df.apply(lambda x: f"lines/{x.subject:04}-{x.line}.jpg", axis=1)
    images, true_transcripts = get_images_and_transcripts(df, 57)

    best_sequence, encoded_image = more_inspects(images, model, processor)

    inspect_attention()

