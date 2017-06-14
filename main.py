import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Conv2D, ELU, Dropout
from keras.utils import plot_model


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

def convert_to_matrix(input_dir):
    image_path_list = glob.glob(os.path.join(input_dir, '*.jpg'))
    for image_path in image_path_list:
            image = cv2.imread

def preprocessing(input_dir, output_dir):
    """
    Preprocess all images with cropping and grayscale then save them to disk.
    """

    assert(os.path.exists(input_dir))
    if not os.path.exists(image_output_dir):
        os.mkdir(image_output_dir)
    image_path_list = glob.glob(os.path.join(input_dir, '*.jpg'))
    for image_path in image_path_list:
        image_name = image_path[len(input_dir) + 1:]
        image = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert to BGR for opencv visualization
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        cropped_image = gray_image[70:137, 0:]
        cv2.imwrite(os.path.join(output_dir, image_name), cropped_image)
        print('preprocessed ', image_name, ' to path ', output_dir)
        # print(gray_image.shape)
        # show_image((1, 1, 1), image_path, cropped_image, 3)  # location, path, image, width?

def simple_test_model(input_shape=(160, 320, 3)):
    """Very simple model to test out flow"""

    # create model
    model = Sequential()

    # normalize pixels to between -1 <= x <= 1
    model.add(Lambda(lambda x: (x / 255.0) * 2 - 1, input_shape=input_shape))

    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")

    return model

if __name__ == "__main__":
    image_input_dir = 'data/IMG'
    image_output_dir = 'data/preproc_IMG'
    preprocessing(image_input_dir, image_output_dir)
