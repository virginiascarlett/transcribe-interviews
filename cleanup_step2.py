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
parser = argparse.ArgumentParser(description="Perform final-pass transcript cleanup using an LLM.")
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
elif args.provider == "litellm":
    from my_litellm import query_LLM

# Get env variables
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
DATA_SUBDIR = os.getenv("DATA_SUBDIR")
if args.provider == "openai":
    LLM_TEST_MODEL = os.getenv("OPENAI_TEST_MODEL")
    LLM_PROD_MODEL = os.getenv("OPENAI_PROD_MODEL")
elif args.provider == "litellm":
    LLM_TEST_MODEL = os.getenv("LITELLM_TEST_MODEL")
    LLM_PROD_MODEL = os.getenv("LITELLM_PROD_MODEL")


model = LLM_TEST_MODEL if args.test else LLM_PROD_MODEL

data_path = Path(DATA_DIR, DATA_SUBDIR)
# Create a list of Path objects
files = sorted((data_path / "diarized_transcripts_somewhat_clean").glob("*.txt"))

INSTRUCTIONS = """
You are a transcript clean-up service. Your job is to take raw
interview transcripts and clean up the punctuation and grammar to
make them more readable. Remove filler words (um, uh, like, you know)
and correct capitalization errors, punctuation errors, and minor typos.
The output should be plain text with no formatting (bold text, bullet points, etc.).
Do not edit the substance of the statements from the transcript.
"""


def clean_data(file):
    results = []

    with open(file, "r") as inF:
        text = inF.read()

    USER_DATA = f"""
    Please clean up this transcript:
    {text}
    """
    answer = query_LLM(INSTRUCTIONS, USER_DATA, model=model)
    if answer.startswith("Error:") or answer.startswith("An unexpected error"):
        print(f"Error processing {file}: {answer}")
        results.append(None)
    else:
        results.append(answer)

    return results

# Run the process
results_list = utils.run_func_w_progbar(
    clean_data,
    [files],
    output_path=data_path,
    output_subdir="diarized_transcripts_clean_final",
    output_basename="final",
    output_extension="txt",
)

utils.report_time(start_time)
