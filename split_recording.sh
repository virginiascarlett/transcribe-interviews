#!/bin/bash

# Load env variables
set -a
source .env
set +a

# Validate command-line argument
if [ -z "$1" ]; then
    echo "Error: Please provide a recording name as an argument." >&2
    echo "Usage: ./split_recording.sh <recording_name>" >&2
    exit 1
fi

# Strip .mp4 extension if present
RECORDING_NAME="$1"
RECORDING_NAME="${RECORDING_NAME%.mp4}"

# Validate that the input MP4 file exists
INPUT_FILE="$DATA_DIR/$RECORDING_NAME.mp4"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE" >&2
    exit 1
fi

# Set up the .tmp_<name> directory
TMP_DIR="$DATA_DIR/.tmp_$RECORDING_NAME"
LOG_FILE="$TMP_DIR/split_recording.log"

# Create the directory if it doesn't exist
if [ ! -d "$TMP_DIR" ]; then
    mkdir -p "$TMP_DIR"
else
    # If directory exists, delete all chunk*.mp4 files inside
    rm -f "$TMP_DIR/chunk"*.mp4
fi

# Proceed with the ffmpeg command
ffmpeg -i "$INPUT_FILE" -c copy -map 0 -segment_time 00:08:00 -f segment "$TMP_DIR/chunk%d.mp4" >> "$LOG_FILE" 2>&1
if [ $? -eq 0 ]; then
    echo "split_recording: success. See $LOG_FILE for details."
else
    echo "split_recording: error. See $LOG_FILE for details." >&2
    exit 1
fi