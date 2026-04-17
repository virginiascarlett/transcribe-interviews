#!/bin/bash

# Run this script to split the recording downloaded from Zoom
# into 8-minute chunks.

# Load env variables
set -a
source .env
set +a

ffmpeg -i $DATA_DIR/$DATA_SUBDIR/recording.mp4 -c copy -map 0 -segment_time 00:08:00 -f segment $DATA_DIR/$1/chunk%d.mp4