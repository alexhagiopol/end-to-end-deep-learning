import glob
import shutil

if __name__ == "__main__":
    img_image_paths = glob.glob('driving_data/IMG/*')
    img1_image_paths = glob.glob('driving_data/IMG1/*')
    destination_path = 'driving_data/ian_images'
    for path in img_image_paths + img1_image_paths:
        shutil.move(path, destination_path)
