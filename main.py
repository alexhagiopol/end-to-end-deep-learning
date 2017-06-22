import utilities
import pickle
import os

if __name__ == "__main__":
    # load data from pickle file
    utilities.preprocess('data', 'images', max_num_measurements=100)

