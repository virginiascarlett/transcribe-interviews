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
diarization_files = sorted((data_path/"diarizations").glob("*.txt"))
transcript_files = sorted((data_path/"transcripts").glob("*.txt"))

litellm.api_base = "https://litellm.dreamlab.ucsb.edu"

# Choose model (Flash is faster/cheaper, Pro is smarter)
# "gemini-3-flash-preview", "gemini-3.1-pro-preview", or "gemini-3.1-pro-preview-customtools"
MODEL_NAME = "litellm_proxy/gemini-3-flash-preview"

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

    try:
        response = completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": USER_DATA}
            ]
        )

        # Extract the text answer
        answer = response.choices[0].message.content
        results.append(answer)

    except Exception as e:
        chunk_num = ''.join(filter(str.isdigit, str(diarization) ))
        print(f"Error processing chunk {chunk_num}")
        results.append(None)

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