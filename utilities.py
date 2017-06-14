import matplotlib.pyplot as plt
import os
import cv2
import glob


def preprocessing(input_dir, output_dir):
    """
    Preprocess all images with cropping and grayscale then save them to disk.
    """
    assert(os.path.exists(input_dir))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    image_path_list = glob.glob(os.path.join(input_dir, '*.jpg'))
    for image_path in image_path_list:
        image_name = image_path[len(input_dir) + 1:]
        image = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert to BGR for opencv visualization
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        cropped_image = gray_image[70:137, 0:]
        cv2.imwrite(os.path.join(output_dir, image_name), cropped_image)
        print('preprocessed ', image_name, ' to path ', output_dir, ' with shape ', cropped_image.shape)


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