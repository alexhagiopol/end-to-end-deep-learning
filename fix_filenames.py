import pandas as pd
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide log file to fix \n Example: python fix_filenames.py my_log.csv")
    else:
        csv_path = sys.argv[1]
        driving_log = pd.read_csv(csv_path, header=None)
        num_records = driving_log.shape[0]
        record_index = 0
        while record_index < num_records:
            center_file_path = driving_log.iloc[record_index, 0]
            left_file_path = driving_log.iloc[record_index, 1]
            right_file_path = driving_log.iloc[record_index, 2]
            print("Fixing record row #", record_index, " of ", num_records, center_file_path, left_file_path,
                  right_file_path)
            driving_log.iloc[record_index, 0] = center_file_path[-34:]
            driving_log.iloc[record_index, 1] = left_file_path[-32:]
            driving_log.iloc[record_index, 2] = right_file_path[-33:]
            record_index += 1
        print("Overwriting CSV file: ", csv_path)
        driving_log.to_csv(csv_path, header=None, index=False)
        print("Done.")

