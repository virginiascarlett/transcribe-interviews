import time
from tqdm import tqdm
from pathlib import Path

def report_time(start_time):
    """Calculates and prints the elapsed time."""
    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    print(f"\nTime taken: {minutes}m {seconds:.2f}s")

def run_func_w_progbar(func, input_files):
    """
    Executes a function and, meanwhile, prints a progress bar.
    input_files is a list containing either 1 or 2 lists of files
    to work on.
    """
    results_list = []
    progress_bar = tqdm(input_files[0])
    for i, data_file in enumerate(progress_bar):
        progress_bar.set_description(f"Processing file {i}")
        if len(input_files) == 1:
            results_list.append(func(data_file))
        elif len(input_files) == 2:
            results_list.append(func(data_file, input_files[1][i]))
            # ^ we also could have written this as:
            # func(input_files[0][i], input_files[1][i])
        else:
            raise ValueError(f"Received {len(input_files)} sets of input files; should be 1 or 2")
    return results_list

