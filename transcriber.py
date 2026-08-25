#!/usr/bin/env python
"""
Transcriber module for converting audio chunks to text using the Whisper model.

This module processes audio chunks from interview recordings and generates
time-stamped transcripts using OpenAI's Whisper model.
"""

import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
import whisper
import utils


class Transcriber:
    """Handles transcription of audio files using the Whisper model."""

    def __init__(self, model_name: str, data_dir: str):
        """
        Initialize the Transcriber.

        Args:
            model_name: Name of the Whisper model to use (e.g., 'tiny', 'base', 'large-v3-turbo')
            data_dir: Base directory for data operations
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.model = whisper.load_model(model_name)

    def transcribe_chunk(self, data_file: str) -> dict:
        """
        Transcribe a single audio chunk.

        Args:
            data_file: Path to the audio file to transcribe

        Returns:
            Dictionary containing transcription results with segments and timing
        """
        # verbose=False to suppress progress output since we use tqdm
        # fp16=False to suppress an annoying warning after it tries and fails to use fp16
        # language=en prevents guessing and suppresses the "language detected" message
        return self.model.transcribe(str(data_file), fp16=False, verbose=False, language="en")

    def transcribe_all(self, recording_name: str) -> list:
        """
        Transcribe all audio chunks for a recording.

        Args:
            recording_name: Name of the recording (basename without .mp4 extension)

        Returns:
            List of transcription results
        """
        # Get files to process
        data_path = Path(self.data_dir, f".tmp_{recording_name}")

        # Verify the .tmp_<name> directory exists
        if not data_path.exists():
            raise FileNotFoundError(f"Directory not found: {data_path}")

        # Create a list of Path objects
        data_files = sorted(data_path.glob("*.mp4"))

        # Ensure all files are chunks
        if not all(f.name.startswith("chunk") for f in data_files):
            raise ValueError(
                "Found an mp4 that is not a chunk in the .tmp_ directory. "
                "Please ensure only chunk*.mp4 files are present in the directory."
            )

        # Do the work
        result_list = utils.run_func_w_progbar(
            self.transcribe_chunk,
            [[str(f) for f in data_files]],
            output_path=data_path,
            output_subdir=None,
            output_basename="whisper_transcript",
            output_extension="txt",
            save_func=self._save_transcription,
        )

        return result_list

    @staticmethod
    def _save_transcription(out_file: str, result: dict) -> None:
        """
        Save transcription results to a file with timestamps.

        Args:
            out_file: Path to output file
            result: Transcription result dictionary from Whisper
        """
        with open(out_file, "w") as outF:
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()
                outF.write(f"[{start:.2f}s - {end:.2f}s] {text}\n")


def load_transcriber_config() -> tuple:
    """
    Load transcriber configuration from environment variables.

    Returns:
        Tuple of (DATA_DIR, WHISPER_TEST_MODEL, WHISPER_PROD_MODEL)
    """
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    whisper_test_model = os.getenv("WHISPER_TEST_MODEL")
    whisper_prod_model = os.getenv("WHISPER_PROD_MODEL")
    return data_dir, whisper_test_model, whisper_prod_model


if __name__ == "__main__":
    start_time = time.perf_counter()

    # Parse command line args
    parser = argparse.ArgumentParser(description="Transcribe audio chunks.")
    parser.add_argument("--name", required=True, help="Recording name (basename without .mp4 extension)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--test", action="store_true", help="Use the test model (WHISPER_TEST_MODEL)")
    group.add_argument("-p", "--prod", action="store_true", help="Use the production model (WHISPER_PROD_MODEL)")
    args = parser.parse_args()

    # Load configuration
    data_dir, whisper_test_model, whisper_prod_model = load_transcriber_config()
    model_name = whisper_test_model if args.test else whisper_prod_model

    # Create transcriber and run
    transcriber = Transcriber(model_name=model_name, data_dir=data_dir)
    transcriber.transcribe_all(args.name)

    utils.report_time(start_time)
