#!/bin/bash

# Load env variables
set -a
source .env
set +a

# Find all mp4 files in $DATA_DIR/$DATA_SUBDIR
mp4_files=($(find "$DATA_DIR/$DATA_SUBDIR" -maxdepth 1 -name "*.mp4"))

# Count the number of mp4 files
num_mp4_files=${#mp4_files[@]}

LOG_FILE="$DATA_DIR/$DATA_SUBDIR/split_recording.log"

if [ $num_mp4_files -eq 1 ]; then

    # Proceed with the ffmpeg command
    ffmpeg -i ${mp4_files[0]} -c copy -map 0 -segment_time 00:08:00 -f segment "$DATA_DIR/$DATA_SUBDIR/chunk%d.mp4" >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        echo "split_recording: success. See $LOG_FILE for details."
    else
        echo "split_recording: error. See $LOG_FILE for details." >&2
    fi
else
    echo "Error: There must be exactly one mp4 file in $DATA_DIR/$DATA_SUBDIR. Found $num_mp4_files files." >&2
    exit 1
fi