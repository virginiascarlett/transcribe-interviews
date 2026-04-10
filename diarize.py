#!/usr/bin/env python
import os
import time
from tqdm import tqdm
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
HF_TOKEN = os.getenv("HF_TOKEN")

# Get command line args
args = utils.get_args()
DATA_SUBDIR = args.data_subdir

# Create a list of file names to process
data_files = utils.get_files(DATA_DIR, DATA_SUBDIR, "wav")

# Suppress warnings - I get a well-known "degrees of freedom is <= 0" that seems safe to ignore.
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")


# Because pyannote.audio uses gated models,
# I had to create a Hugging Face account
# and sign their agreement to get a token.
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)

progress_bar = tqdm(data_files)
for counter, data_file in enumerate(progress_bar):
    out_path = os.path.join(DATA_DIR, DATA_SUBDIR, f"diarization{counter}.txt")
    progress_bar.set_description(f"Processing {data_file}")

    diarization = pipeline(data_file)
    annotation = diarization.speaker_diarization
    diarization_output = []
    for segment, track, speaker in annotation.itertracks(yield_label=True):
        diarization_output.append(
            f"[{segment.start:.1f}s - {segment.end:.1f}s] {speaker}"
        )

    with open(out_path, "w") as outF:
        outF.write("\n".join(diarization_output))

utils.report_time(start_time)
