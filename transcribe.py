#!/usr/bin/env python
import os
import glob
import argparse
from dotenv import load_dotenv
import whisper

# Loads variables from .env into os.environ
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_subdir",
    help="the subdirectory under data/ containing the recordings you want to process; e.g. conversation1"
    )
args = parser.parse_args()

DATA_SUBDIR = args.data_subdir

data_files = sorted(
    [ data_file for data_file in glob.iglob(os.path.join(DATA_DIR, DATA_SUBDIR, '*.mp4')) ]
    )

# Load the model (options: tiny, base, small, medium, large, turbo)
# I use base for testing and medium for production
model = whisper.load_model("medium")
counter = 0
stop = len(data_files)
while counter < stop:
    for data_file in data_files:
        result = model.transcribe(data_file)
        outF = open(
            os.path.join(DATA_DIR, DATA_SUBDIR, f"transcript{counter}.txt"),
            'w')
        for segment in result['segments']:
            outF.write(segment['text'].strip() + '\n')
        outF.close()
        counter += 1
