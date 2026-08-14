#!/bin/bash

# Run this script to convert mp4 files from Zoom to .wav for pyannote.

# Load env variables
set -a
source .env
set +a

LOG_FILE="$DATA_DIR/$DATA_SUBDIR/convert_to_wav.log"
errors=0

for file in "$DATA_DIR"/$DATA_SUBDIR/chunk*.mp4; do
    # Extract filename without extension for the output
    filename=$(basename "$file" .mp4)
    ffmpeg -i "$file" "$DATA_DIR/$DATA_SUBDIR/${filename}.wav" >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        errors=$((errors + 1))
    fi
done

if [ $errors -eq 0 ]; then
    echo "convert_to_wav: success. See $LOG_FILE for details."
else
    echo "convert_to_wav: $errors file(s) failed to convert. See $LOG_FILE for details." >&2
fi