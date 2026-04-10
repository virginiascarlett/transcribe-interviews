import os
import glob
import argparse
import time

def get_args():
    """Standardized argument parser for the project."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_subdir",
        help="The subdirectory under data/ containing the recordings (e.g., conversation1)",
    )
    return parser.parse_args()

def get_files(data_dir, subdir, extension):
    """Finds and sorts files with a specific extension in the target directory."""
    pattern = os.path.join(data_dir, subdir, f"*.{extension}")
    return sorted(glob.glob(pattern))

def report_time(start_time):
    """Calculates and prints the elapsed time."""
    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    print(f"\nTime taken: {minutes}m {seconds:.2f}s")