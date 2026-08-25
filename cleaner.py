#!/usr/bin/env python
"""
Cleaner module for LLM-based transcript cleanup.

This module provides two-step cleanup of merged transcripts:
1. Step 1: Remove redundant speaker labels and consolidate speech
2. Step 2: Final cleanup of grammar, punctuation, and filler words
"""

import os
import sys
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import utils


class Cleaner:
    """Handles LLM-based transcript cleanup."""

    INSTRUCTIONS_STEP1 = """
You have been given a diarized interview excerpt. Your goal is
to improve its readability by removing redundant speaker labels.
Rules:
    A speaker label (e.g., "SPEAKER_01:") should only appear when the speaker changes.
    If the same speaker continues speaking across multiple consecutive lines, omit the label for all lines after the first one.
    Do not change, summarize, or correct the text of the transcript. Keep the words and punctuation exactly as they are.
    Consolidate line breaks so that each speaker's speech is one paragraph.
"""

    INSTRUCTIONS_STEP2 = """
You are a transcript clean-up service. Your job is to take raw
interview transcripts and clean up the punctuation and grammar to
make them more readable. Remove filler words (um, uh, like, you know)
and correct capitalization errors, punctuation errors, and minor typos.
The output should be plain text with no formatting (bold text, bullet points, etc.).
Do not edit the substance of the statements from the transcript.
Do not remove the speaker labels.
"""

    def __init__(self, llm_model: str, data_dir: str, query_llm_func):
        """
        Initialize the Cleaner.

        Args:
            llm_model: Name of the LLM model to use
            data_dir: Base directory for data operations
            query_llm_func: Function to query the LLM (from my_openai or my_litellm)
        """
        self.llm_model = llm_model
        self.data_dir = data_dir
        self.query_llm_func = query_llm_func

    def clean_chunk(self, data_file: str, step: int = 1) -> list:
        """
        Clean a single transcript chunk.

        Args:
            data_file: Path to the file to clean
            step: Cleanup step (1 or 2)

        Returns:
            List containing the cleaned result
        """
        results = []

        with open(data_file, "r") as inF:
            transcript_text = inF.read()

        if step == 1:
            instructions = self.INSTRUCTIONS_STEP1
            user_data = f"""
Please clean up the speaker labels in the following transcript:
<transcript>
{transcript_text}
</transcript>
"""
        elif step == 2:
            instructions = self.INSTRUCTIONS_STEP2
            user_data = f"""
Please clean up this transcript:
{transcript_text}
"""
        else:
            raise ValueError(f"Unknown cleanup step: {step}")

        answer = self.query_llm_func(instructions, user_data, model=self.llm_model)
        if answer.startswith("Error:") or answer.startswith("An unexpected error"):
            print(f"Error processing {data_file}: {answer}")
            results.append(None)
        else:
            results.append(answer)

        return results

    def clean_step1(self, recording_name: str) -> list:
        """
        Perform first-pass cleanup (remove redundant speaker labels).

        Args:
            recording_name: Name of the recording (basename without .mp4 extension)

        Returns:
            List of cleanup results
        """
        # Construct the temp directory path
        temp_dir = Path(self.data_dir, f".tmp_{recording_name}")

        # Validate that the directory exists
        if not temp_dir.exists():
            raise FileNotFoundError(f"Directory {temp_dir} does not exist")

        if not temp_dir.is_dir():
            raise NotADirectoryError(f"{temp_dir} is not a directory")

        # Validate that merged0.txt exists
        merged0_path = temp_dir / "merged0.txt"
        if not merged0_path.exists():
            raise FileNotFoundError(f"{merged0_path} does not exist")

        # Create a list of Path objects for merged files
        data_files = sorted([f for f in temp_dir.glob("merged[0-9]*.txt")])

        # Run the process
        results_list = utils.run_func_w_progbar(
            lambda f: self.clean_chunk(f, step=1),
            [data_files],
            output_path=temp_dir,
            output_subdir=None,
            output_basename="merged_semiclean",
            output_extension="txt",
        )

        return results_list

    def clean_step2(self, recording_name: str) -> list:
        """
        Perform final cleanup (grammar, punctuation, filler words).

        Args:
            recording_name: Name of the recording (basename without .mp4 extension)

        Returns:
            List of cleanup results
        """
        # Construct the temp directory path
        temp_dir = Path(self.data_dir, f".tmp_{recording_name}")

        # Validate that the directory exists
        if not temp_dir.exists():
            raise FileNotFoundError(f"Directory {temp_dir} does not exist")

        if not temp_dir.is_dir():
            raise NotADirectoryError(f"{temp_dir} is not a directory")

        # Validate that merged_semiclean0.txt exists
        merged_semiclean0_path = temp_dir / "merged_semiclean0.txt"
        if not merged_semiclean0_path.exists():
            raise FileNotFoundError(f"{merged_semiclean0_path} does not exist")

        # Create a list of Path objects for merged_semiclean files
        files = sorted([f for f in temp_dir.glob("merged_semiclean[0-9]*.txt")])

        # Run the process
        results_list = utils.run_func_w_progbar(
            lambda f: self.clean_chunk(f, step=2),
            [files],
            output_path=temp_dir,
            output_subdir=None,
            output_basename="merged_clean",
            output_extension="txt",
        )

        return results_list


def load_cleaner_config(provider: str) -> tuple:
    """
    Load cleaner configuration from environment variables.

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Clean up transcripts using an LLM.")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the recording (basename without .mp4 extension)",
    )
    step_group = parser.add_mutually_exclusive_group(required=True)
    step_group.add_argument("--step1", action="store_true", help="Perform first-pass cleanup")
    step_group.add_argument("--step2", action="store_true", help="Perform final-pass cleanup")
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

    start_time = time.perf_counter()

    # Load configuration
    data_dir, llm_test_model, llm_prod_model, query_llm_func = load_cleaner_config(args.provider)
    model = llm_test_model if args.test else llm_prod_model

    # Create cleaner and run
    cleaner = Cleaner(llm_model=model, data_dir=data_dir, query_llm_func=query_llm_func)

    if args.step1:
        cleaner.clean_step1(args.name)
    elif args.step2:
        cleaner.clean_step2(args.name)

    utils.report_time(start_time)
