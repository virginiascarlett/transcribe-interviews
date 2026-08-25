#!/bin/bash

# Run this script to convert mp4 files from Zoom to .wav for pyannote.
# Usage: ./convert_to_wav.sh <recording_name>
# Example: ./convert_to_wav.sh my_interview

if [ $# -ne 1 ]; then
    echo "Usage: $0 <recording_name>" >&2
    exit 1
fi

RECORDING_NAME="$1"

# Load env variables
set -a
source .env
set +a

TMP_DIR="$DATA_DIR/.tmp_${RECORDING_NAME}"
LOG_FILE="$TMP_DIR/convert_to_wav.log"
errors=0

for file in "$TMP_DIR"/chunk*.mp4; do
    # Extract filename without extension for the output
    filename=$(basename "$file" .mp4)
    ffmpeg -i "$file" "$TMP_DIR/${filename}.wav" >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        errors=$((errors + 1))
    fi
done

if [ $errors -eq 0 ]; then
    echo "convert_to_wav: success. See $LOG_FILE for details."
else
    echo "convert_to_wav: $errors file(s) failed to convert. See $LOG_FILE for details." >&2
fi