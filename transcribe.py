#!/usr/bin/env python
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
import whisper
import utils

# Start the clock - we'll report how long the script took to run
start_time = time.perf_counter()

####### EDIT ME: WHISPER MODEL #######
# Options: tiny, base, small, medium, large, turbo
# I use tiny or base for testing and medium for production
# Have not tried large-v3-turbo but it's supposed to be good
# and only use a little more memory than medium
model = whisper.load_model("tiny")
######################################

# Get env variables
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

# Get command line args
args = utils.get_args()
DATA_SUBDIR = args.data_subdir

# Create a list of file names to process
data_files = utils.get_files(DATA_DIR, DATA_SUBDIR, "mp4")

progress_bar = tqdm(data_files)
for counter, data_file in enumerate(progress_bar):
    out_path = os.path.join(DATA_DIR, DATA_SUBDIR, f"transcript{counter}.txt")
    progress_bar.set_description(f"Processing {data_file}")

    # verbose=False to suppress progress output since we are using tqdm
    # fp16=False, to suppress an annoying warning after it tries and fails to use fp16
    # language=en prevents guessing and suppresses the "language detected" message
    result = model.transcribe(data_file, fp16=False, verbose=False, language="en")

    with open(out_path, "w") as outF:
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            outF.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")

utils.report_time(start_time)
