#!/usr/bin/env python
import os
import argparse
from pathlib import Path
import time
import subprocess
from dotenv import load_dotenv
import utils

# Start the clock - we'll report how long the script took to run
start_time = time.perf_counter()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--name", type=str, required=True, help="The name of the recording")
model_group = parser.add_mutually_exclusive_group(required=True)
model_group.add_argument("-t", "--test", action="store_true", help="Run the pipeline in test mode")
model_group.add_argument("-p", "--prod", action="store_true", help="Run the pipeline in production mode")
parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "litellm"],
        required=True,
        help="The provider to use (must be either openai or litellm)."
    )
args = parser.parse_args()

# Get env variables
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")

# Strip .mp4 extension from name argument if needed.
recording_name = args.name.rstrip('.mp4') if args.name.endswith('.mp4') else args.name

# Run split_recording using the name argument like so: ./split_recording.sh <my_recording_name>
print(f"Running split_recording.sh for {recording_name}...")
subprocess.run(["./split_recording.sh", recording_name], check=True)
print("split_recording.sh completed successfully.\n")

# Run transcribe.py using the arguments passed to this script (test/prod and name)
print(f"Running transcribe.py for {recording_name}...")
transcribe_args = ["python", "transcribe.py", "--name", recording_name]
if args.test:
    transcribe_args.append("--test")
else:
    transcribe_args.append("--prod")
subprocess.run(transcribe_args, check=True)
print("transcribe.py completed successfully.\n")

# Run convert_to_wav.sh using the name argument like so: ./convert_to_wav.sh <my_recording_name>
print(f"Running convert_to_wav.sh for {recording_name}...")
subprocess.run(["./convert_to_wav.sh", recording_name], check=True)
print("convert_to_wav.sh completed successfully.\n")

# Run diarize.py using the name argument passed to this script
print(f"Running diarize.py for {recording_name}...")
subprocess.run(["python", "diarize.py", "--name", recording_name], check=True)
print("diarize.py completed successfully.\n")

# Report total time taken
utils.report_time(start_time)