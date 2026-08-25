#!/usr/bin/env python
"""
This script transcribes audio from MP4 files using the Whisper
model by OpenAI. It processes audio chunks of an interview
recording and generates time-stamped transcripts.
Make sure your environment variables in .env are up-to-date.
"""

import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
import whisper
import utils

# Start the clock - we'll report how long the script took to run
start_time = time.perf_counter()

# Parse command line args
parser = argparse.ArgumentParser(description="Transcribe audio chunks.")
parser.add_argument('--name', required=True, help="Recording name (basename without .mp4 extension)")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-t', '--test', action='store_true', help="Use the test model (WHISPER_TEST_MODEL)")
group.add_argument('-p', '--prod', action='store_true', help="Use the production model (WHISPER_PROD_MODEL)")
args = parser.parse_args()

# Get env variables
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
WHISPER_TEST_MODEL = os.getenv("WHISPER_TEST_MODEL")
WHISPER_PROD_MODEL = os.getenv("WHISPER_PROD_MODEL")
WHISPER_MODEL = WHISPER_TEST_MODEL if args.test else WHISPER_PROD_MODEL

# Model options: tiny, base, small, medium, large-v3-turbo
# I use tiny or base for testing and large-v3-turbo for production
model = whisper.load_model(WHISPER_MODEL)


def transcribe(data_file):
    # verbose=False to suppress progress output since we are using tqdm
    # fp16=False, to suppress an annoying warning after it tries and fails to use fp16
    # language=en prevents guessing and suppresses the "language detected" message
    return model.transcribe(str(data_file), fp16=False, verbose=False, language="en")


# Get files to process
data_path = Path(DATA_DIR, f".tmp_{args.name}")

# Verify the .tmp_<name> directory exists
if not data_path.exists():
    raise FileNotFoundError(f"Directory not found: {data_path}")

# Create a list of Path objects
data_files = sorted(data_path.glob("*.mp4"))

# Ensure all files are chunks
if all(f.name.startswith("chunk") for f in data_files):
    pass
else:
    raise ValueError("""
                     Found an mp4 that is not a chunk in the .tmp_ directory.
                     Please ensure only chunk*.mp4 files are present in the directory.
                     """)


def save_transcription(out_file, result):
    with open(out_file, "w") as outF:
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            outF.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")


# Do the work
result_list = utils.run_func_w_progbar(
    transcribe,
    [[str(f) for f in data_files]],
    output_path=data_path,
    output_subdir=None,
    output_basename="whisper_transcript",
    output_extension="txt",
    save_func=save_transcription,
)

utils.report_time(start_time)
