#!/usr/bin/env python
"""
Diarizer module for performing speaker diarization on audio files.

This module identifies and labels different speakers in audio files using
the pyannote.audio speaker diarization pipeline.
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import torch
import utils


class Diarizer:
    """Handles speaker diarization of audio files using pyannote."""

    def __init__(self, hf_token: str, data_dir: str):
        """
        Initialize the Diarizer.

        Args:
            hf_token: Hugging Face API token for accessing gated models
            data_dir: Base directory for data operations
        """
        self.hf_token = hf_token
        self.data_dir = data_dir
        self._pipeline = None  # Lazy load the pipeline

    @property
    def pipeline(self) -> Pipeline:
        """
        Lazy-load the diarization pipeline on first access.

        Returns:
            The pyannote speaker diarization pipeline
        """
        if self._pipeline is None:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", token=self.hf_token
            )
        return self._pipeline

    def diarize_chunk(self, data_file: str) -> list:
        """
        Perform speaker diarization on a single audio file.

        Args:
            data_file: Path to the audio file to diarize

        Returns:
            List of diarization entries with timestamps and speaker labels
        """
        diarization = self.pipeline(data_file)
        annotation = diarization.speaker_diarization
        diarization_output = []
        for segment, track, speaker in annotation.itertracks(yield_label=True):
            diarization_output.append(f"[{segment.start:.1f}s - {segment.end:.1f}s] {speaker}")
        return diarization_output

    def diarize_all(self, recording_name: str) -> list:
        """
        Perform speaker diarization on all audio files for a recording.

        Args:
            recording_name: Name of the recording (basename without .mp4 extension)

        Returns:
            List of diarization results
        """
        # Get files to process
        data_path = Path(self.data_dir, f".tmp_{recording_name}")

        # Validate that the directory exists and contains .wav files
        if not data_path.exists():
            raise FileNotFoundError(f"Directory {data_path} does not exist")

        if not data_path.is_dir():
            raise NotADirectoryError(f"{data_path} is not a directory")

        # Create a list of Path objects
        data_files = sorted(data_path.glob("*.wav"))

        if not data_files:
            raise FileNotFoundError(f"No .wav files found in {data_path}")

        print("Dear user, this step is slow! Please be patient. See README for more details.")

        # Do the work
        result_list = utils.run_func_w_progbar(
            self.diarize_chunk,
            [[str(f) for f in data_files]],
            output_path=data_path,
            output_subdir=None,
            output_basename="diarization",
            output_extension="txt",
        )

        return result_list


def load_diarizer_config() -> tuple:
    """
    Load diarizer configuration from environment variables.

    Returns:
        Tuple of (DATA_DIR, HF_TOKEN)
    """
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    hf_token = os.getenv("HF_TOKEN")
    return data_dir, hf_token


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Perform speaker diarization on .wav files.")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the recording (basename without .mp4 extension)",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()

    # Suppress warnings - well-known "degrees of freedom is <= 0" warning that seems safe to ignore
    warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

    # Load configuration
    data_dir, hf_token = load_diarizer_config()

    # Create diarizer and run
    diarizer = Diarizer(hf_token=hf_token, data_dir=data_dir)
    diarizer.diarize_all(args.name)

    utils.report_time(start_time)
