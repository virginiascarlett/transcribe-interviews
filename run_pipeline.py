#!/usr/bin/env python
"""
Pipeline orchestrator for interview transcription and processing.

This script coordinates the entire workflow:
1. Splitting recording into chunks
2. Transcribing chunks with Whisper
3. Converting to WAV format for diarization
4. Performing speaker diarization
5. Merging transcription with speaker identification
6. Two-pass cleanup using LLM
7. Concatenating final results
"""

import os
import argparse
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
import utils

from transcriber import Transcriber, load_transcriber_config
from diarizer import Diarizer, load_diarizer_config
from merger import Merger, load_merger_config
from cleaner import Cleaner, load_cleaner_config


def run_pipeline():
    """Run the complete interview transcription pipeline."""

    # Start the clock - we'll report how long the script took to run
    start_time = time.perf_counter()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the full interview transcription and processing pipeline.")
    parser.add_argument("--name", type=str, required=True, help="The name of the recording")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("-t", "--test", action="store_true", help="Run the pipeline in test mode")
    model_group.add_argument("-p", "--prod", action="store_true", help="Run the pipeline in production mode")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "litellm"],
        required=True,
        help="The provider to use (must be either openai or litellm).",
    )
    args = parser.parse_args()

    # Get env variables
    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR")

    # Strip .mp4 extension from name argument if needed.
    recording_name = args.name.rstrip(".mp4") if args.name.endswith(".mp4") else args.name

    # Step 1: Split recording into chunks
    print(f"Splitting {recording_name} into chunks...")
    subprocess.run(["./split_recording.sh", recording_name], check=True)

    # Step 2: Run whisper transcription
    print(f"Running whisper transcription for {recording_name}...")
    transcriber_data_dir, whisper_test_model, whisper_prod_model = load_transcriber_config()
    whisper_model = whisper_test_model if args.test else whisper_prod_model
    transcriber = Transcriber(model_name=whisper_model, data_dir=transcriber_data_dir)
    transcriber.transcribe_all(recording_name)

    # Step 3: Convert from mp4 to wav for pyannote
    print(f"Converting {recording_name} from mp4 to wav for pyannote...")
    subprocess.run(["./convert_to_wav.sh", recording_name], check=True)

    # Step 4: Perform diarization
    print(f"Performing diarization for {recording_name}...")
    diarizer_data_dir, hf_token = load_diarizer_config()
    diarizer = Diarizer(hf_token=hf_token, data_dir=diarizer_data_dir)
    diarizer.diarize_all(recording_name)

    # Step 5: Merge transcript and diarization
    print(f"Merging transcript and diarization for {recording_name}...")
    merger_data_dir, merger_test_model, merger_prod_model, merger_query_llm = load_merger_config(args.provider)
    merger_model = merger_test_model if args.test else merger_prod_model
    merger = Merger(llm_model=merger_model, data_dir=merger_data_dir, query_llm_func=merger_query_llm)
    merger.merge_all(recording_name)

    # Step 6: First-pass cleanup (remove redundant speaker labels)
    print(f"Performing first-pass cleanup for {recording_name}...")
    cleaner_data_dir, cleaner_test_model, cleaner_prod_model, cleaner_query_llm = load_cleaner_config(
        args.provider
    )
    cleaner_model = cleaner_test_model if args.test else cleaner_prod_model
    cleaner = Cleaner(llm_model=cleaner_model, data_dir=cleaner_data_dir, query_llm_func=cleaner_query_llm)
    cleaner.clean_step1(recording_name)

    # Step 7: Final cleanup (grammar, punctuation, filler words)
    print(f"Performing final cleanup for {recording_name}...")
    cleaner.clean_step2(recording_name)

    # Step 8: Concatenate chunks for final result
    print(f"Concatenating chunks for {recording_name}...")
    subprocess.run(["./concat_final_results.sh", recording_name], check=True)

    print(f"Pipeline complete. Delete {DATA_DIR}/.tmp_{recording_name} if you are happy with the results.")

    # Report total time taken
    utils.report_time(start_time)


if __name__ == "__main__":
    run_pipeline()
