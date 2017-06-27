import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import glob


def preprocess(image_matrix):
    image_matrix_gray = cv2.cvtColor(image_matrix, cv2.COLOR_RGB2GRAY)
    image_matrix_cropped = image_matrix_gray[70:137, 0:]
    image_matrix_cropped_normalized = image_matrix_cropped / 255 - 0.5
    return image_matrix_cropped_normalized


def batch_preprocess(data_dir_name, image_subdir_names, l_r_correction=0.2, debug=False, max_num_measurements=None, pickle_file_name='pickle_data.p'):
    """
    Preprocess all images and measurements then save them to disk in Keras-compatible format.
    # + numbers go right, - numbers go left. Thus for left camera we correct right and for right camera we collect left.
    """
    driving_log = None
    for image_subdir_name in image_subdir_names:
        image_input_dir = os.path.join(data_dir_name, image_subdir_name)
        assert(os.path.exists(image_input_dir))
        current_log_path = os.path.join(image_input_dir, 'log')
        assert(os.path.exists(current_log_path))
        current_log_file_list = glob.glob(os.path.join(current_log_path, '*.csv'))
        assert(len(current_log_file_list) == 1)
        current_log_file = current_log_file_list[0]
        if driving_log is not None:
            driving_log = pd.read_csv(current_log_file, header=None)
        else:
            current_driving_log = pd.read_csv(current_log_file)
            driving_log = pd.concat([driving_log, current_driving_log])
    if max_num_measurements:
        num_measurements = max_num_measurements
    else:
        num_measurements = driving_log.shape[0]
    num_images = num_measurements * 6  # * 6 because of left center and right image for each entry and their flipped versions.
    y_train = np.zeros(6 * num_measurements)  # we 6X the number of measurements because we have 3 cameras and we flip each view to generate 6 (images, steering) pairs for each measurement
    X_train = np.zeros((num_images, 67, 320))
    measurement_index = 0
    while measurement_index < num_measurements:
        datum_index = measurement_index * 6
        # CENTER CAMERA IMAGE
        y_train[datum_index] = driving_log.iloc[measurement_index, 3]  # center image steering value added to dataset
        center_image_filename = driving_log.iloc[measurement_index, 0][4:]  # get rid of "IMG/" in data log
        center_image_path = os.path.join(image_input_dir, center_image_filename)
        center_image_matrix = cv2.imread(center_image_path)
        preprocessed_center_image_matrix = preprocess(center_image_matrix)
        X_train[datum_index, :, :] = preprocessed_center_image_matrix  # center image matrix added to dataset
        # LEFT CAMERA IMAGE
        y_train[datum_index + 1] = driving_log.iloc[measurement_index, 3] + l_r_correction  # left image steering value added to dataset
        left_image_filename = driving_log.iloc[measurement_index, 1][5:]  # get rid of " IMG/" in data log
        left_image_path = os.path.join(image_input_dir, left_image_filename)
        left_image_matrix = cv2.imread(left_image_path)
        preprocessed_left_image_matrix = preprocess(left_image_matrix)
        X_train[datum_index + 1, :, :] = preprocessed_left_image_matrix  # left image matrix added to dataset
        # RIGHT CAMERA IMAGE
        y_train[datum_index + 2] = driving_log.iloc[measurement_index, 3] - l_r_correction  # right image steering value added to dataset
        right_image_filename = driving_log.iloc[measurement_index, 2][5:]  # get rid of " IMG/" in data log
        right_image_path = os.path.join(image_input_dir, right_image_filename)
        right_image_matrix = cv2.imread(right_image_path)
        preprocessed_right_image_matrix = preprocess(right_image_matrix)
        X_train[datum_index + 2, :, :] = preprocessed_right_image_matrix  # right image matrix added to dataset
        # FLIPPED CENTER CAMERA IMAGE
        flipped_center = cv2.flip(preprocessed_center_image_matrix, flipCode=1)
        y_train[datum_index + 3] = y_train[datum_index]*-1
        X_train[datum_index + 3, :, :] = flipped_center
        # FLIPPED LEFT CAMERA IMAGE
        flipped_left = cv2.flip(preprocessed_left_image_matrix, flipCode=1)
        y_train[datum_index + 4] = y_train[datum_index + 1]*-1
        X_train[datum_index + 4, :, :] = flipped_left
        # FLIPPED RIGHT CAMERA IMAGE
        flipped_right = cv2.flip(preprocessed_right_image_matrix, flipCode=1)
        y_train[datum_index + 5] = y_train[datum_index + 2]*-1
        X_train[datum_index + 5, :, :] = flipped_right
        measurement_index += 1
        if debug:
            show_image((2, 3, 1), "left " + str(y_train[datum_index + 1]), preprocessed_left_image_matrix)
            show_image((2, 3, 2), "center " + str(y_train[datum_index]), preprocessed_center_image_matrix)
            show_image((2, 3, 3), "right " + str(y_train[datum_index + 2]), preprocessed_right_image_matrix)
            show_image((2, 3, 4), "left flipped " + str(y_train[datum_index + 4]), flipped_left)
            show_image((2, 3, 5), "center flipped " + str(y_train[datum_index + 3]), flipped_center)
            show_image((2, 3, 6), "right flipped " + str(y_train[datum_index + 5]), flipped_right)
            plt.show()
            plt.close()
        print('processed ', measurement_index, ' of ', num_measurements, ' measurements. Images:', center_image_filename, ' ', left_image_filename, ' ', right_image_filename)
    print('Saving processed data to pickle file in ', data_dir_name, ' directory ...')
    pickle_data = {'features': X_train, 'labels': y_train}
    pickle.dump(pickle_data, open(os.path.join(data_dir_name, pickle_file_name), "wb"), protocol=4)  # protocol=4 allows file sizes > 4GB
    print("Done.")


def show_image(location, title, img, width=None):
    if width is not None:
        plt.figure(figsize=(width, width))
    plt.subplot(*location)
    plt.title(title, fontsize=8)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    if width is not None:
        plt.show()
        plt.close()
