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
parser = argparse.ArgumentParser(description="Merge diarization and transcript files using an LLM.")
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
    from my_openai import query_LLM
    LLM_TEST_MODEL = os.getenv("OPENAI_TEST_MODEL")
    LLM_PROD_MODEL = os.getenv("OPENAI_PROD_MODEL")
elif args.provider == "litellm":
    from my_litellm import query_LLM
    LLM_TEST_MODEL = os.getenv("LITELLM_TEST_MODEL")
    LLM_PROD_MODEL = os.getenv("LITELLM_PROD_MODEL")

model = LLM_TEST_MODEL if args.test else LLM_PROD_MODEL

# Create a list of Path objects
diarization_files = sorted((data_path/"diarizations").glob("*.txt"))
transcript_files = sorted((data_path/"transcripts").glob("*.txt"))

INSTRUCTIONS = """
You have been given two artifacts from an interview excerpt: one is a transcript
of what was said, and the other is a record of who spoke when. Please
merge the two into one file using the timestamps provided.
The result should be formatted as speaker: statement, without timestamps, like this:
SPEAKER_00: Thanks for joining us today.
SPEAKER_01: Sure, happy to be here.
SPEAKER_00: To get started, tell me about your role in this project.
Do not edit the statements from the transcript.
"""


def merge_data(diarization, transcript):
    results = []

    with open(diarization, "r") as inF:
        diarization_text = inF.read()

    with open(transcript, "r") as inF:
        transcript_text = inF.read()

    # Use xml tags to demarcate the start and end of each file
    USER_DATA = f"""
        Please merge these two documents:
        <transcript>
        {transcript_text}
        </transcript>

        <diarization>
        {diarization_text}
        </diarization>
        """

    answer = query_LLM(INSTRUCTIONS, USER_DATA, model=model)
    if answer.startswith("Error:") or answer.startswith("An unexpected error"):
        chunk_num = ''.join(filter(str.isdigit, str(diarization)))
        print(f"Error processing chunk {chunk_num}: {answer}")
        results.append(None)
    else:
        results.append(answer)

    return results

# Run the process
results_list = utils.run_func_w_progbar(
    merge_data,
    [diarization_files, transcript_files],
    output_path=data_path,
    output_subdir="diarized_transcripts_raw",
    output_basename="merged",
    output_extension="txt"
)

utils.report_time(start_time)