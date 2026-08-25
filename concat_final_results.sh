#!/bin/bash

# Run this script to concatenate the final merged and cleaned transcript files.

# Load env variables
set -a
source .env
set +a

# Validate command-line argument
if [ -z "$1" ]; then
    echo "Error: Please provide a recording name as an argument." >&2
    echo "Usage: $0 <recording_name>" >&2
    exit 1
fi

RECORDING_NAME="$1"

# Validate that the input transcripts exist
TMP_DIR="$DATA_DIR/.tmp_${RECORDING_NAME}"

INPUT_FILE="$TMP_DIR/merged_clean0.txt"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE" >&2
    exit 1
fi

for file in "$TMP_DIR"/merged_clean*.txt; do
    # Extract filename without extension for the output
    filename=$(basename "$file" .txt)
    cat "$file"
    echo
done > "$DATA_DIR/${RECORDING_NAME}_final_transcript.txt"
