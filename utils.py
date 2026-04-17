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

def run_func_w_progbar(func, input_files, output_path=None, output_subdir=None, output_basename=None, output_extension=None, save_func=None):
    """
    Executes a function and, meanwhile, prints a progress bar.
    input_files is a list containing either 1 or 2 lists of files
    to work on.
    If output parameters are provided, it saves results incrementally.
    """
    results_list = []
    progress_bar = tqdm(input_files[0])
    for i, data_file in enumerate(progress_bar):
        progress_bar.set_description(f"Processing file {i}")
        if len(input_files) == 1:
            result = func(data_file)
        elif len(input_files) == 2:
            result = func(data_file, input_files[1][i])
        else:
            raise ValueError(f"Received {len(input_files)} sets of input files; should be 1 or 2")
        
        results_list.append(result)

        if output_path:
            out_file = get_out_file_path(output_path, output_subdir, output_basename, i, output_extension)
            if save_func:
                save_func(out_file, result)
            else:
                with open(out_file, "w") as outF:
                    if isinstance(result, list):
                        outF.write("\n".join(result))
                    else:
                        outF.write(str(result))
                        
    return results_list

def get_out_file_path(path_w_subdir, sub_subdir, basename, index, extension):
    """Constructs the full path to an output file, e.g. dummy_data/conversation1/transcripts/transcript0/txt."""
    out_dir = path_w_subdir / sub_subdir
    out_file = out_dir / f"{basename}{index}.{extension}"
    # parents=True creates any missing parent directories in the path
    # exist_ok=True means don't overwrite the folder if it's already there
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_file
