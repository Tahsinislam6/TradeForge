import glob
import os


def clear_external_files(directory: str, pattern: str = "*"):
    """Delete files in a directory that match a glob pattern.

    Usage:
        clear_external_files(indicators_dir)
        clear_external_files(indicators_dir, "*_{trial_number}.csv")

    This is useful for cleaning up generated indicator/output CSV files
    before or after a test run.
    """
    # The pattern is joined to the directory, so callers only pass the folder
    # and the glob they want to clear.
    for file_path in glob.glob(os.path.join(directory, pattern)):
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"Warning: Could not delete {file_path}. It is likely locked by another trial. Continuing.")
            except FileNotFoundError:
                continue