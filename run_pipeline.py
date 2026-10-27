#!/usr/bin/env python
"""
Pipeline orchestrator for interview transcription and processing.

This script coordinates the entire workflow:
1. Splitting recording into chunks (split)
2. Transcribing chunks with Whisper (transcribe)
3. Converting to WAV format for diarization (convert)
4. Performing speaker diarization (diarize)
5. Merging transcription with speaker identification (merge)
6. First-pass cleanup (remove redundant speaker labels) (cleanup1)
7. Final cleanup (grammar, punctuation, filler words) (cleanup2)
8. Concatenating final results (concat)
"""

import os
import argparse
import subprocess
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
import utils

from transcriber import Transcriber, load_transcriber_config
from diarizer import Diarizer, load_diarizer_config
from merger import Merger, load_merger_config
from cleaner import Cleaner, load_cleaner_config


# Step definitions with metadata
STEPS = {
    1: {
        "nickname": "split",
        "description": "Split recording into chunks",
        "dependencies": [],
    },
    2: {
        "nickname": "transcribe",
        "description": "Run Whisper transcription",
        "dependencies": [1],
    },
    3: {
        "nickname": "convert",
        "description": "Convert MP4 to WAV",
        "dependencies": [1],
    },
    4: {
        "nickname": "diarize",
        "description": "Perform speaker diarization",
        "dependencies": [3],
    },
    5: {
        "nickname": "merge",
        "description": "Merge transcript with diarization",
        "dependencies": [2, 4],
    },
    6: {
        "nickname": "cleanup1",
        "description": "First-pass cleanup (remove redundant speaker labels)",
        "dependencies": [5],
    },
    7: {
        "nickname": "cleanup2",
        "description": "Final cleanup (grammar, punctuation, filler words)",
        "dependencies": [6],
    },
    8: {
        "nickname": "concat",
        "description": "Concatenate chunks for final result",
        "dependencies": [7],
    },
}


def parse_step_input(step_input):
    """
    Convert step input (number string or nickname) to step number.
    Returns the step number (1-8) or raises ValueError if invalid.
    """
    # Try parsing as a number first
    try:
        step_num = int(step_input)
        if 1 <= step_num <= 8:
            return step_num
        else:
            raise ValueError(f"Step number must be between 1 and 8, got {step_num}")
    except ValueError as e:
        if "invalid literal" not in str(e):
            raise

    # Try matching as a nickname
    for step_num, step_info in STEPS.items():
        if step_info["nickname"] == step_input.lower():
            return step_num

    # If we get here, the input was invalid
    available = ", ".join(
        f"{num} ({info['nickname']})" for num, info in STEPS.items()
    )
    raise ValueError(
        f"Invalid step: '{step_input}'. Available steps: {available}"
    )


def print_dependency_warning(step_num):
    """Print a warning about dependencies for a given step."""
    step_info = STEPS[step_num]
    if not step_info["dependencies"]:
        return

    deps_str = ", ".join(
        f"Step {dep} ({STEPS[dep]['nickname']})"
        for dep in step_info["dependencies"]
    )
    print(
        f"\nWARNING: Running step {step_num} ({step_info['nickname']}) only. "
        f"This depends on:\n- {deps_str.replace(', ', chr(10) + '- ')}\n"
        f"Make sure these steps have been completed.\n"
    )


def get_steps_to_run(start_step=None, run_only_step=None):
    """
    Determine which steps to run.
    Returns a list of step numbers to execute.
    """
    if run_only_step is not None:
        return [run_only_step]
    elif start_step is not None:
        return list(range(start_step, 9))
    else:
        return list(range(1, 9))


def step_split(recording_name):
    """Step 1: Split recording into chunks."""
    print(f"Splitting {recording_name} into chunks...")
    subprocess.run(["./split_recording.sh", recording_name], check=True)


def step_transcribe(recording_name, args, transcriber_config):
    """Step 2: Run Whisper transcription."""
    print(f"Running whisper transcription for {recording_name}...")
    transcriber_data_dir, whisper_test_model, whisper_prod_model = transcriber_config
    whisper_model = whisper_test_model if args.test else whisper_prod_model
    transcriber = Transcriber(model_name=whisper_model, data_dir=transcriber_data_dir)
    transcriber.transcribe_all(recording_name)


def step_convert(recording_name):
    """Step 3: Convert from mp4 to wav for pyannote."""
    print(f"Converting {recording_name} from mp4 to wav for pyannote...")
    subprocess.run(["./convert_to_wav.sh", recording_name], check=True)


def step_diarize(recording_name, diarizer_config):
    """Step 4: Perform diarization."""
    print(f"Performing diarization for {recording_name}...")
    diarizer_data_dir, hf_token = diarizer_config
    diarizer = Diarizer(hf_token=hf_token, data_dir=diarizer_data_dir)
    diarizer.diarize_all(recording_name)


def step_merge(recording_name, args, merger_config):
    """Step 5: Merge transcript and diarization."""
    print(f"Merging transcript and diarization for {recording_name}...")
    merger_data_dir, merger_test_model, merger_prod_model, merger_query_llm = merger_config
    merger_model = merger_test_model if args.test else merger_prod_model
    merger = Merger(
        llm_model=merger_model,
        data_dir=merger_data_dir,
        query_llm_func=merger_query_llm,
        num_interviewers=args.num_interviewers,
        num_interviewees=args.num_interviewees,
    )
    merger.merge_all(recording_name)


def step_cleanup1(recording_name, args, cleaner_config):
    """Step 6: First-pass cleanup (remove redundant speaker labels)."""
    print(f"Performing first-pass cleanup for {recording_name}...")
    cleaner_data_dir, cleaner_test_model, cleaner_prod_model, cleaner_query_llm = cleaner_config
    cleaner_model = cleaner_test_model if args.test else cleaner_prod_model
    cleaner = Cleaner(llm_model=cleaner_model, data_dir=cleaner_data_dir, query_llm_func=cleaner_query_llm)
    cleaner.clean_step1(recording_name)


def step_cleanup2(recording_name, args, cleaner_config):
    """Step 7: Final cleanup (grammar, punctuation, filler words)."""
    print(f"Performing final cleanup for {recording_name}...")
    cleaner_data_dir, cleaner_test_model, cleaner_prod_model, cleaner_query_llm = cleaner_config
    cleaner_model = cleaner_test_model if args.test else cleaner_prod_model
    cleaner = Cleaner(llm_model=cleaner_model, data_dir=cleaner_data_dir, query_llm_func=cleaner_query_llm)
    cleaner.clean_step2(recording_name)


def step_concat(recording_name, data_dir):
    """Step 8: Concatenate chunks for final result."""
    print(f"Concatenating chunks for {recording_name}...")
    subprocess.run(["./concat_final_results.sh", recording_name], check=True)


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
    parser.add_argument(
        "--num_interviewers",
        type=int,
        required=False,
        help="Optional command line arg to specify number of interviewers",
    )
    parser.add_argument(
        "--num_interviewees",
        type=int,
        required=False,
        help="Optional command line arg to specify number of interviewees",
    )

    # Step selection arguments (mutually exclusive)
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--run-only",
        type=str,
        help="Run only this step (specify by number 1-8 or nickname)",
    )
    step_group.add_argument(
        "--start-from",
        type=str,
        help="Start pipeline from this step (specify by number 1-8 or nickname)",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(1)

    # Get env variables
    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR")

    # Strip .mp4 extension from name argument if needed.
    recording_name = args.name.rstrip(".mp4") if args.name.endswith(".mp4") else args.name

    # Parse and validate step selection
    run_only_step = None
    start_from_step = None

    if args.run_only:
        try:
            run_only_step = parse_step_input(args.run_only)
            print_dependency_warning(run_only_step)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if args.start_from:
        try:
            start_from_step = parse_step_input(args.start_from)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Determine which steps to run
    steps_to_run = get_steps_to_run(start_from_step, run_only_step)

    # Load configs that might be needed
    transcriber_config = load_transcriber_config()
    diarizer_config = load_diarizer_config()
    merger_config = load_merger_config(args.provider)
    cleaner_config = load_cleaner_config(args.provider)

    # Execute the selected steps
    for step_num in steps_to_run:
        if step_num == 1:
            step_split(recording_name)
        elif step_num == 2:
            step_transcribe(recording_name, args, transcriber_config)
        elif step_num == 3:
            step_convert(recording_name)
        elif step_num == 4:
            step_diarize(recording_name, diarizer_config)
        elif step_num == 5:
            step_merge(recording_name, args, merger_config)
        elif step_num == 6:
            step_cleanup1(recording_name, args, cleaner_config)
        elif step_num == 7:
            step_cleanup2(recording_name, args, cleaner_config)
        elif step_num == 8:
            step_concat(recording_name, DATA_DIR)

    # Only show the completion message if running the full pipeline
    if run_only_step is None and start_from_step is None:
        print(f"Pipeline complete. Delete {DATA_DIR}/.tmp_{recording_name} if you are happy with the results.")
    else:
        step_range = f"step(s) {steps_to_run[0]}" if len(steps_to_run) == 1 else f"steps {steps_to_run[0]}-{steps_to_run[-1]}"
        print(f"Completed {step_range}.")

    # Report total time taken
    utils.report_time(start_time)


if __name__ == "__main__":
    run_pipeline()
