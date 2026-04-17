#!/usr/bin/env python
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import warnings
from pyannote.audio import Pipeline
import torch
import utils

# Start the clock - we'll report how long the script took to run
start_time = time.perf_counter()

# Get env variables
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
DATA_SUBDIR = os.getenv("DATA_SUBDIR")
HF_TOKEN = os.getenv("HF_TOKEN")
# Suppress warnings - I get a well-known "degrees of freedom is <= 0" warning that seems safe to ignore.
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

def diarize(data_file):
    # Because pyannote.audio uses gated models,
    # I had to create a Hugging Face account
    # and sign their agreement to get a token.
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
    diarization = pipeline(data_file)
    annotation = diarization.speaker_diarization
    diarization_output = []
    for segment, track, speaker in annotation.itertracks(yield_label=True):
        diarization_output.append(
            f"[{segment.start:.1f}s - {segment.end:.1f}s] {speaker}"
        )
    return diarization_output

# Get files to process
data_path = Path(DATA_DIR, DATA_SUBDIR)
# Create a list of Path objects
data_files = sorted(data_path.glob("*.wav"))

# Do the work
result_list = utils.run_func_w_progbar(
    diarize, 
    [[str(f) for f in data_files]],
    output_path=data_path,
    output_subdir="diarizations",
    output_basename="diarization",
    output_extension="txt"
)

utils.report_time(start_time)
