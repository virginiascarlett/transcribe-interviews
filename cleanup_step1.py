#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from dotenv import load_dotenv
import utils

# Start the clock - we'll report how long the script took to run
start_time = time.perf_counter()

# Get env variables
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
DATA_SUBDIR = os.getenv("DATA_SUBDIR")

data_path = Path(DATA_DIR, DATA_SUBDIR)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Clean up speaker labels in diarized transcripts using an LLM.")
model_group = parser.add_mutually_exclusive_group(required=True)
model_group.add_argument("-t", "--test", action="store_true", help="Use the test LLM model (LLM_TEST_MODEL)")
model_group.add_argument("-p", "--prod", action="store_true", help="Use the production LLM model (LLM_PROD_MODEL)")
parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "litellm"],
        required=True,
        help="The provider to use (must be either openai or litellm)."
    )
args = parser.parse_args()

if args.provider == "openai":
    from openai import query_LLM
    LLM_TEST_MODEL = os.getenv("OPENAI_TEST_MODEL")
    LLM_PROD_MODEL = os.getenv("OPENAI_PROD_MODEL")
elif args.provider == "litellm":
    from my_litellm import query_LLM
    LLM_TEST_MODEL = os.getenv("LITELLM_TEST_MODEL")
    LLM_PROD_MODEL = os.getenv("LITELLM_PROD_MODEL")

model = LLM_TEST_MODEL if args.test else LLM_PROD_MODEL

# Create a list of Path objects
data_files = sorted((data_path / "diarized_transcripts_raw").glob("*.txt"))

INSTRUCTIONS = """
You have been given a diarized interview excerpt. Your goal is
to improve its readability by removing redundant speaker labels.
Rules:
    A speaker label (e.g., "SPEAKER_01:") should only appear when the speaker changes.
    If the same speaker continues speaking across multiple consecutive lines, omit the label for all lines after the first one.
    Do not change, summarize, or correct the text of the transcript. Keep the words and punctuation exactly as they are.
    Consolidate line breaks so that each speaker's speech is one paragraph.
"""

def clean_data(data_file):
    results = []

    with open(data_file, "r") as inF:
        transcript_text = inF.read()

    # Use xml tags to demarcate the start and end of each file
    USER_DATA = f"""
        Please clean up the speaker labels in the following transcript:
        <transcript>
        {transcript_text}
        </transcript>
        """

    answer = query_LLM(INSTRUCTIONS, USER_DATA, model=model)
    if answer.startswith("Error:") or answer.startswith("An unexpected error"):
        print(f"Error processing {data_file}: {answer}")
        results.append(None)
    else:
        results.append(answer)

    return results

# Run the process
results_list = utils.run_func_w_progbar(
    clean_data,
    [[str(f) for f in data_files]],
    output_path=data_path,
    output_subdir="diarized_transcripts_somewhat_clean",
    output_basename="chunk",
    output_extension="txt"
)

utils.report_time(start_time)