#!/usr/bin/env python
import os
from pathlib import Path
import time
from dotenv import load_dotenv
import litellm
from litellm import completion
import utils

# Start the clock - we'll report how long the script took to run
start_time = time.perf_counter()

# Get env variables
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
DATA_SUBDIR = os.getenv("DATA_SUBDIR")
LITELLM_PROXY_API_KEY = os.getenv("LITELLM_PROXY_API_KEY")

data_path = Path(DATA_DIR, DATA_SUBDIR)
# Create a list of Path objects
files = sorted((data_path / "diarized_transcripts_clean").glob("*.txt"))

litellm.api_base = "https://litellm.dreamlab.ucsb.edu"

# Choose model (Flash is faster/cheaper, Pro is smarter)
# "gemini-3-flash-preview", "gemini-3.1-pro-preview", or "gemini-3.1-pro-preview-customtools"
MODEL_NAME = "litellm_proxy/gemini-3.1-pro-preview"

INSTRUCTIONS = """
You are a transcript clean-up service. Your job is to take raw
interview transcripts and clean up the punctuation and grammar to
make them more readable. Remove filler words (um, uh, like, you know)
and correct capitalization errors, punctuation errors, and minor typos.
The output should be plain text with no formatting (bold text, bullet points, etc.).
Do not edit the substance of the statements from the transcript.
"""


def merge_data(file):
    results = []

    with open(file, "r") as inF:
        text = inF.read()

    # Use xml tags to demarcate the start and end of each file
    USER_DATA = f"""
        Please clean up this transcript:
        {text}
        """

    try:
        response = completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": USER_DATA},
            ],
        )

        # Extract the text answer
        answer = response.choices[0].message.content
        results.append(answer)

    except Exception as e:
        chunk_num = "".join(filter(str.isdigit, str(file)))
        print(f"Error processing chunk {chunk_num}")
        results.append(None)

    return results


# Run the process
results_list = utils.run_func_w_progbar(
    merge_data,
    [files],
    output_path=data_path,
    output_subdir="diarized_transcripts_clean_final",
    output_basename="final",
    output_extension="txt",
)

utils.report_time(start_time)
