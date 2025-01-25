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


def visualize_images_and_transcripts(images, transcripts):
    for img in images:
        display(img)

    for transcript in transcripts:
        print(transcript)


def transcribe_images(all_images, model, processor):
    """
    Transcribe a batch of images using an OCR model.

    Args:
        all_images: a list of PIL images.
        model: the model to do image-to-token ids
        processor: the processor which maps token ids to text

    Returns:
        a list of the transcribed text.
    """
    pixel_values = processor(images=all_images, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values.to(device))
    decoded_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded_text


def clean_string(input_string):
    """
    Clean string prior to comparison

    Args:
        input_string (str): the input string

    Returns:
        (str) a cleaned string, lowercase, alphabetical characters only, no double spaces
    """

    # Convert all characters to lowercase
    lowercase_string = input_string.lower()

    # Remove non-alphabetic characters
    alpha_string = re.sub(r'[^a-z\s]', '', lowercase_string)

    # Remove double spaces and start and end spaces
    return re.sub(r'\s+', ' ', alpha_string).strip()


def calculate_mismatch(estimated_text, reference_text):
    """
    Calculate mismatch (character and word error rates) between estimated and true text.

    Args:
        estimated_text: a list of strings
        reference_text: a list of strings

    Returns:
        A tuple, (CER and WER)
    """
    # Lowercase the text and remove special characters for the comparison
    estimated_text = [clean_string(x) for x in estimated_text]
    reference_text = [clean_string(x) for x in reference_text]

    # Calculate the character error rate and word error rates. They should be
    # raw floats, not tensors.
    cer = fm.char_error_rate(estimated_text, reference_text).item()
    wer = fm.word_error_rate(estimated_text, reference_text).item()
    return cer, wer


def calculate_all_mismatch(df, model, processor):
    """
    Calculate CER and WER for all subjects in a dataset

    Args:
        df: a dataframe containing information about images and transcripts
        model: an image-to-text model
        processor: a processor object

    Returns:
        a list of dictionaries containing a per-subject breakdown of the
        results
    """
    subjects = df.subject.unique().tolist()

    results = []

    # Calculate CER and WER for all subjects
    for subject in tqdm.tqdm(subjects):

        # Load images and labels for a given subject
        images, true_transcripts = get_images_and_transcripts(df, subject)

        # Transcribe the images to text
        transcribed_text = transcribe_images(images, model, processor)

        # Calculate the CER and WER
        cer, wer = calculate_mismatch(transcribed_text, true_transcripts)

        results.append({
            'subject': subject,
            'cer': cer,
            'wer': wer,
        })
    return results


def calculate_mean_max_cer(df_results):
    """
    Calculate the mean character-error-rate across subjects as
    well as the maximum (that is, the OOD risk).

    Args:
        df_results: a dataframe containing results

    Returns:
        A tuple, (mean_cer, max_cer)
    """

    # Calculate the mean CER across test subjects.
    mean_subjects = df_results.cer.mean()

    # Calculate the max CER across test subjects.
    max_subjects = df_results.cer.max()
    return mean_subjects, max_subjects


if __name__ == '__main__':
    df = pd.read_csv('transcripts.csv')
    df['filename'] = df.apply(lambda x: f"lines/{x.subject:04}-{x.line}.jpg", axis=1)
    images, true_transcripts = get_images_and_transcripts(df, 52)
    visualize_images_and_transcripts(images, true_transcripts)

    model, processor = load_pretrained_model()

    transcribed_text = transcribe_images(images, model, processor)

    cer, wer = calculate_mismatch(transcribed_text, true_transcripts)
    assert isinstance(cer, float)
    print(cer, wer)

    results = calculate_all_mismatch(df, model, processor)
    df_results = pd.DataFrame(results)
    print(df_results)
    print("A subject that's harder to read")
    images, true_transcripts = get_images_and_transcripts(df, 57)
    visualize_images_and_transcripts(images, true_transcripts)

    mean_subjects, max_subjects = calculate_mean_max_cer(df_results)
    print(mean_subjects, max_subjects)


