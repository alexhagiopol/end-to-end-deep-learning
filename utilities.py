import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import pickle


def preprocessing(data_dir_name, image_subdir_name, l_r_correction=0.2, debug=False):
    """
    Preprocess all images and measurements then save them to disk in Keras-compatible format.
    # + numbers go right, - numbers go left. Thus for left camera we correct right and for right camera we collect left.
    """
    image_dir = os.path.join(data_dir_name, image_subdir_name)
    assert(os.path.exists(image_dir))
    image_input_dir = 'data/IMG'
    driving_log = pd.read_csv('data/driving_log.csv')
    num_measurements = driving_log.shape[0]
    num_images = num_measurements * 6  # * 6 because of left center and right image for each entry and their flipped versions.
    y_train = np.zeros(6 * num_measurements)  # we 6X the number of measurements because we have 3 cameras and we flip each view to generate 6 (images, steering) pairs for each measurement
    X_train = np.zeros((67, 320, num_images))
    measurement_index = 0
    while measurement_index < num_measurements:
        datum_index = measurement_index * 6
        # CENTER CAMERA IMAGE
        y_train[datum_index] = driving_log.iloc[measurement_index, 3]  # center image steering value added to dataset
        center_image_filename = driving_log.iloc[measurement_index, 0][4:]  # get rid of "IMG/" in data log
        center_image_path = os.path.join(image_input_dir, center_image_filename)
        center_image_matrix = cv2.imread(center_image_path)
        center_image_matrix_gray = cv2.cvtColor(center_image_matrix, cv2.COLOR_RGB2GRAY)
        center_image_matrix_cropped = center_image_matrix_gray[70:137, 0:]
        X_train[:, :, datum_index] = center_image_matrix_cropped  # center image matrix added to dataset
        # LEFT CAMERA IMAGE
        y_train[datum_index + 1] = driving_log.iloc[measurement_index, 3] + l_r_correction  # left image steering value added to dataset
        left_image_filename = driving_log.iloc[measurement_index, 1][5:]  # get rid of " IMG/" in data log
        left_image_path = os.path.join(image_input_dir, left_image_filename)
        left_image_matrix = cv2.imread(left_image_path)
        left_image_matrix_gray = cv2.cvtColor(left_image_matrix, cv2.COLOR_RGB2GRAY)
        left_image_matrix_cropped = left_image_matrix_gray[70:137, 0:]
        X_train[:, :, datum_index + 1] = left_image_matrix_cropped  # left image matrix added to dataset
        # RIGHT CAMERA IMAGE
        y_train[datum_index + 2] = driving_log.iloc[measurement_index, 3] - l_r_correction  # right image steering value added to dataset
        right_image_filename = driving_log.iloc[measurement_index, 2][5:]  # get rid of " IMG/" in data log
        right_image_path = os.path.join(image_input_dir, right_image_filename)
        right_image_matrix = cv2.imread(right_image_path)
        right_image_matrix_gray = cv2.cvtColor(right_image_matrix, cv2.COLOR_RGB2GRAY)
        right_image_matrix_cropped = right_image_matrix_gray[70:137, 0:]
        X_train[:, :, datum_index + 2] = right_image_matrix_cropped  # right image matrix added to dataset
        # FLIPPED CENTER CAMERA IMAGE
        flipped_center = cv2.flip(center_image_matrix_cropped, flipCode=1)
        y_train[datum_index + 3] = y_train[datum_index]*-1
        X_train[:, :, datum_index + 3] = flipped_center
        # FLIPPED LEFT CAMERA IMAGE
        flipped_left = cv2.flip(left_image_matrix_cropped, flipCode=1)
        y_train[datum_index + 4] = y_train[datum_index + 1]*-1
        X_train[:, :, datum_index + 4] = flipped_left
        # FLIPPED RIGHT CAMERA IMAGE
        flipped_right = cv2.flip(right_image_matrix_cropped, flipCode=1)
        y_train[datum_index + 5] = y_train[datum_index + 2]*-1
        X_train[:, :, datum_index + 5] = flipped_right
        measurement_index += 1
        if debug:
            show_image((2, 3, 1), "left " + str(y_train[datum_index + 1]), left_image_matrix_cropped)
            show_image((2, 3, 2), "center " + str(y_train[datum_index]), center_image_matrix_cropped)
            show_image((2, 3, 3), "right " + str(y_train[datum_index + 2]), right_image_matrix_cropped)
            show_image((2, 3, 4), "left flipped " + str(y_train[datum_index + 4]), flipped_left)
            show_image((2, 3, 5), "center flipped " + str(y_train[datum_index + 3]), flipped_center)
            show_image((2, 3, 6), "right flipped " + str(y_train[datum_index + 5]), flipped_right)
            plt.show()
            plt.close()
        print('processed ', measurement_index, ' of ', num_measurements, ' measurements. Images:', center_image_filename, ' ', left_image_filename, ' ', right_image_filename)
    print('Saving processed data to pickle file in ', data_dir_name, ' directory ...')
    pickle_data = {'features': X_train, 'labels': y_train}
    pickle.dump(pickle_data, open(os.path.join(data_dir_name, "pickle_data.p"), "wb"), protocol=4)  # protocol=4 allows file sizes > 4GB
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
