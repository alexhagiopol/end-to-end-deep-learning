import sys
import utilities

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide log file to randomize \n Example: python fix_filenames.py my_log.csv")
        sys.exit()
    csv_path = sys.argv[1]
    utilities.randomize_dataset_csv(csv_path)
