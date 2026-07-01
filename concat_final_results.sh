#!/bin/bash

# Run this script to concatenate the final merged and cleaned transcript files.

# Load env variables
set -a
source .env
set +a

for f in $DATA_DIR/$DATA_SUBDIR/diarized_transcripts_clean_final/final*.txt; do cat "$f"; echo "\n"; done > $DATA_DIR/$DATA_SUBDIR/final_transcript.txt