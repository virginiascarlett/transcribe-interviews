#!/usr/bin/env python
import os
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import whisper
import utils

# Start the clock - we'll report how long the script took to run
start_time = time.perf_counter()



# Get env variables
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
DATA_SUBDIR = os.getenv("DATA_SUBDIR")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
# Model options: tiny, base, small, medium, large-v3-turbo
# I use tiny or base for testing and large-v3-turbo for production
model = whisper.load_model(WHISPER_MODEL)

data_path = Path(DATA_DIR, DATA_SUBDIR)
# Create a list of Path objects
data_files = sorted(data_path.glob("*.mp4"))
# Don't process the original recording. Only process chunks.
data_files = [f for f in data_files if f.name != "recording.mp4"]
if all(f.name.startswith('chunk') for f in data_files):
    pass
else:
    raise ValueError("""
                     Found an mp4 that is not called recording.mp4 but is also not a chunk.
                     Please make sure your original recording is named recording.mp4, and
                     don't keep any additional mp4s in your data subdir besides the recording
                     and the chunks.
                     """)

def transcribe(data_file):
    # verbose=False to suppress progress output since we are using tqdm
    # fp16=False, to suppress an annoying warning after it tries and fails to use fp16
    # language=en prevents guessing and suppresses the "language detected" message
    return model.transcribe(str(data_file), fp16=False, verbose=False, language="en")

result_list = utils.run_func_w_progbar(transcribe, [[str(f) for f in data_files]])

for i, result in enumerate(result_list):
    out_path = data_path / f"transcript{i}.txt"
    # parents=True creates any missing parent directories in the path
    # exist_ok=True means don't overwrite the folder if it's already there
    data_path.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as outF:
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            outF.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")

utils.report_time(start_time)
