#!/usr/bin/env python
import os
import argparse
from pathlib import Path
import time
from dotenv import load_dotenv
import utils

# Start the clock - we'll report how long the script took to run
start_time = time.perf_counter()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="")
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
DATA_SUBDIR = os.getenv("DATA_SUBDIR")

# Find all .mp4 files in DATA_DIR/DATA_SUBDIR. If there's more than one, or if there are none, throw an error.

#