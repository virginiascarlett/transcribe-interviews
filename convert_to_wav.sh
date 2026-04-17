#!/bin/bash

# Run this script to convert mp4 files from Zoom to .wav for pyannote.

# Load env variables
set -a
source .env
set +a

for file in "$DATA_DIR"/$DATA_SUBDIR/chunk*.mp4; do
    # Extract filename without extension for the output
    filename=$(basename "$file" .mp4)
    ffmpeg -i "$file" $DATA_DIR/$1/${filename}.wav
done