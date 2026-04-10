#!/usr/bin/env python
import os
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import torch
import utils

# Get env variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)

diarization = pipeline('dummy_data/conversation1/test_multiple_speakers.mp3')

annotation = diarization.speaker_diarization
diarization_output = []
for segment, track, speaker in annotation.itertracks(yield_label=True):
    diarization_output.append(
        f"[{segment.start:.1f}s - {segment.end:.1f}s] {speaker}"
    )

with open('dummy_data/conversation1/diarization.txt', "w") as outF:
    outF.write("\n".join(diarization_output))
