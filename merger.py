#!/usr/bin/env python
"""
Merger module for combining diarization and transcript data using an LLM.

This module merges speaker identification from diarization with the
transcribed text to create speaker-attributed transcripts.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import utils


class Merger:
    """Handles merging of diarization and transcript data using an LLM."""

    INSTRUCTIONS = """
You have been given two artifacts from an interview excerpt: one is a transcript
of what was said, and the other is a record of who spoke when. Please
merge the two into one file using the timestamps provided.
The result should be formatted as speaker: statement, without timestamps, like this:
SPEAKER_00: Thanks for joining us today.
SPEAKER_01: Sure, happy to be here.
SPEAKER_00: To get started, tell me about your role in this project.
Do not edit the statements from the transcript.
"""

    def __init__(self, llm_model: str, data_dir: str, query_llm_func):
        """
        Initialize the Merger.

        Args:
            llm_model: Name of the LLM model to use
            data_dir: Base directory for data operations
            query_llm_func: Function to query the LLM (from my_openai or my_litellm)
        """
        self.llm_model = llm_model
        self.data_dir = data_dir
        self.query_llm_func = query_llm_func

    def merge_chunk(self, diarization_file: str, transcript_file: str) -> list:
        """
        Merge diarization and transcript for a single chunk.

        Args:
            diarization_file: Path to the diarization file
            transcript_file: Path to the transcript file

        Returns:
            List containing the merged result
        """
        results = []

        with open(diarization_file, "r") as inF:
            diarization_text = inF.read()

        with open(transcript_file, "r") as inF:
            transcript_text = inF.read()

        # Use xml tags to demarcate the start and end of each file
        user_data = f"""
Please merge these two documents:
<transcript>
{transcript_text}
</transcript>

<diarization>
{diarization_text}
</diarization>
"""

        answer = self.query_llm_func(self.INSTRUCTIONS, user_data, model=self.llm_model)
        if answer.startswith("Error:") or answer.startswith("An unexpected error"):
            chunk_num = "".join(filter(str.isdigit, str(diarization_file)))
            print(f"Error processing chunk {chunk_num}: {answer}")
            results.append(None)
        else:
            results.append(answer)

        return results

    def merge_all(self, recording_name: str) -> list:
        """
        Merge all diarization and transcript chunks for a recording.

        Args:
            recording_name: Name of the recording (basename without .mp4 extension)

        Returns:
            List of merge results
        """
        # Construct data_path
        data_path = Path(self.data_dir) / f".tmp_{recording_name}"

        # Validate that required files exist
        diarization_file = data_path / "diarization0.txt"
        transcript_file = data_path / "whisper_transcript0.txt"

        if not diarization_file.exists():
            raise FileNotFoundError(f"{diarization_file} does not exist")

        if not transcript_file.exists():
            raise FileNotFoundError(f"{transcript_file} does not exist")

        # Create a list of Path objects
        diarization_files = sorted(data_path.glob("diarization*.txt"))
        transcript_files = sorted(data_path.glob("whisper_transcript*.txt"))

        # Run the process
        results_list = utils.run_func_w_progbar(
            self.merge_chunk,
            [diarization_files, transcript_files],
            output_path=data_path,
            output_subdir=None,
            output_basename="merged",
            output_extension="txt",
        )

        return results_list


def load_merger_config(provider: str) -> tuple:
    """
    Load merger configuration from environment variables.

    Args:
        provider: LLM provider ('openai' or 'litellm')

    Returns:
        Tuple of (DATA_DIR, LLM_TEST_MODEL, LLM_PROD_MODEL, query_llm_func)
    """
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")

    if provider == "openai":
        from my_openai import query_LLM

        query_llm_func = query_LLM
        llm_test_model = os.getenv("OPENAI_TEST_MODEL")
        llm_prod_model = os.getenv("OPENAI_PROD_MODEL")
    elif provider == "litellm":
        from my_litellm import query_LLM

        query_llm_func = query_LLM
        llm_test_model = os.getenv("LITELLM_TEST_MODEL")
        llm_prod_model = os.getenv("LITELLM_PROD_MODEL")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return data_dir, llm_test_model, llm_prod_model, query_llm_func


if __name__ == "__main__":
    start_time = time.perf_counter()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Merge diarization and transcript files using an LLM.")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Basename of the recording (without .mp4 extension).",
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("-t", "--test", action="store_true", help="Use the test LLM model (LLM_TEST_MODEL)")
    model_group.add_argument("-p", "--prod", action="store_true", help="Use the production LLM model (LLM_PROD_MODEL)")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "litellm"],
        required=True,
        help="The provider to use (must be either openai or litellm).",
    )
    args = parser.parse_args()

    # Load configuration
    data_dir, llm_test_model, llm_prod_model, query_llm_func = load_merger_config(args.provider)
    model = llm_test_model if args.test else llm_prod_model

    # Create merger and run
    merger = Merger(llm_model=model, data_dir=data_dir, query_llm_func=query_llm_func)
    merger.merge_all(args.name)

    utils.report_time(start_time)
