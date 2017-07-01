import glob
import shutil

if __name__ == "__main__":
    img_image_paths = glob.glob('data/IMG0/*')
    img1_image_paths = glob.glob('data/IMG1/*')
    destination_path = 'data/combined'
    for path in img_image_paths + img1_image_paths:
        shutil.move(path, destination_path)
