#!/bin/bash

# Run this script to split the recording downloaded from Zoom
# into 8-minute chunks.

# Load env variables
set -a
source .env
set +a

LOG_FILE="$DATA_DIR/$DATA_SUBDIR/split_recording.log"

ffmpeg -i "$DATA_DIR/$DATA_SUBDIR/recording.mp4" -c copy -map 0 -segment_time 00:08:00 -f segment "$DATA_DIR/$DATA_SUBDIR/chunk%d.mp4" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "split_recording: success. See $LOG_FILE for details."
else
    echo "split_recording: error. See $LOG_FILE for details." >&2
fi