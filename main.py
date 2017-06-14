import cv2
import glob
import os
import numpy as np
import csv
import glob
import utilities
import pandas as pd
import pickle

if __name__ == "__main__":
    image_input_dir = 'data/IMG'
    image_preproc_dir = 'data/preproc_IMG'
    driving_log = pd.read_csv('data/driving_log.csv')
    num_measurements = driving_log.shape[0]
    num_images = num_measurements * 3  # * 3 because of left center and right image for each entry.
    y_train = np.zeros(3 * num_measurements)  # we triple the number of measurements because we have 3 cameras
    X_train = np.zeros((67, 320, num_images))
    measurement_index = 0
    while measurement_index < num_measurements:
        datum_index = measurement_index * 3
        y_train[datum_index] = driving_log.iloc[measurement_index, 3]
        y_train[datum_index + 1] = driving_log.iloc[measurement_index, 3]
        y_train[datum_index + 2] = driving_log.iloc[measurement_index, 3]

        center_image_filename = driving_log.iloc[measurement_index, 0][4:]  # get rid of "IMG/" in data log
        center_image_path = os.path.join(image_preproc_dir, center_image_filename)
        center_image_matrix = cv2.imread(center_image_path)[:, :, 0]
        X_train[:, :, datum_index] = center_image_matrix

        left_image_filename = driving_log.iloc[measurement_index, 1][5:]  # get rid of " IMG/" in data log
        left_image_path = os.path.join(image_preproc_dir, left_image_filename)
        left_image_matrix = cv2.imread(left_image_path)[:, :, 0]
        X_train[:, :, datum_index + 1] = left_image_matrix

        right_image_filename = driving_log.iloc[measurement_index, 2][5:]  # get rid of " IMG/" in data log
        right_image_path = os.path.join(image_preproc_dir, right_image_filename)
        right_image_matrix = cv2.imread(right_image_path)[:, :, 0]
        X_train[:, :, datum_index + 2] = right_image_matrix

        measurement_index += 1
        print('processed ', center_image_filename, ' ', left_image_filename, ' ', right_image_filename)

    pickle_data = {'features': X_train, 'labels': y_train}
    pickle.dump(pickle_data, open(os.path.join(image_preproc_dir, "pickle_data.p"), "wb"))
