#!/usr/bin/env python
import os
from pathlib import Path
from tqdm import tqdm
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
data_files = sorted((data_path / "diarized_transcripts_raw").glob("*.txt"))

litellm.api_base = "https://litellm.dreamlab.ucsb.edu"

# Choose model (Flash is faster/cheaper, Pro is smarter)
# "gemini-3-flash-preview", "gemini-3.1-pro-preview", or "gemini-3.1-pro-preview-customtools"
MODEL_NAME = "litellm_proxy/gemini-3-flash-preview"

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
        print(f"Error processing {data_file}")
        results.append(None)

    return results

# Run the process
results_list = utils.run_func_w_progbar(
    clean_data, 
    [[str(f) for f in data_files]],
    output_path=data_path,
    output_subdir="diarized_transcripts_clean",
    output_basename="chunk",
    output_extension="txt"
)

utils.report_time(start_time)