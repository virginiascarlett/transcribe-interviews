#!/usr/bin/env python
import os
from pathlib import Path
from dotenv import load_dotenv
import litellm
from litellm import completion
import utils

# Get env variables
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
LITELLM_PROXY_API_KEY = os.getenv("LITELLM_PROXY_API_KEY")

data_path = Path(DATA_DIR) / utils.get_args().data_subdir
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
for i in range(len(diarization_files)):
    final_output = merge_data(diarization_files[i], transcript_files[i])
    out_dir = data_path / "diarized_transcripts_raw"
    out_path = out_dir / f"result{i}.txt"
    # parents=True creates any missing parent directories in the path
    # exist_ok=True means don't overwrite the folder if it's already there
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as outF:
        for i, res in enumerate(final_output):
            outF.write(f"\n--- Result {i+1} ---\n{res}")