import pandas as pd
import sys

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Please provide log file to fix \n Example: python fix_filenames.py my_log.csv")
    else:
        csv_path = sys.argv[1]
