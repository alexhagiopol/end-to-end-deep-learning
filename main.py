import architecture
import utilities
import sys
import numpy as np
import os
from sklearn.utils import shuffle


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Incorrect syntax.")
        print("Example syntax: python main.py udacity_dataset model.h5 3000")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_input_dir = sys.argv[1]
    model_name = sys.argv[2]
    batch_size = int(sys.argv[3])
    measurement_index = 0
    dataset_log = utilities.get_dataset_from_csv(image_input_dir)
    dataset_size = dataset_log.shape[0]
    # prepare master set of validation sets
    X_valid_master = None
    y_valid_master = None
    model = architecture.nvidia_model()
    while measurement_index < dataset_size:
        end_index = measurement_index + batch_size
        if end_index < dataset_size:
            print("Processing from index", measurement_index, "to index", end_index)
            preprocessed_batch = utilities.batch_preprocess(image_input_dir, measurement_range=(measurement_index, end_index))
        else:
            print("Processing from index", measurement_index, "to index", dataset_size)
            preprocessed_batch = utilities.batch_preprocess(image_input_dir, measurement_range=(measurement_index, None))
        X_batch = preprocessed_batch['features']
        y_batch = preprocessed_batch['labels']
        print("Done preprocessing.")
        print("features data shape", X_batch.shape)
        print("labels data shape", y_batch.shape)
        X_batch_shuffled, y_batch_shuffled = shuffle(X_batch, y_batch, random_state=0)
        X_train_batch = X_batch_shuffled[0:int(0.8 * X_batch_shuffled.shape[0]), :, :, :]
        y_train_batch = y_batch_shuffled[0:int(0.8 * y_batch_shuffled.shape[0])]
        X_valid_batch = X_batch_shuffled[int(0.8 * X_batch_shuffled.shape[0]):, :, :, :]
        y_valid_batch = y_batch_shuffled[int(0.8 * y_batch_shuffled.shape[0]):]
        if X_valid_master is None and y_valid_master is None:
            X_valid_master = X_valid_batch
            y_valid_master = y_valid_batch
            print("Initialized master validation set. Shape =", X_valid_master.shape)
        else:
            X_valid_master = np.concatenate((X_valid_master, X_valid_batch), axis=0)
            y_valid_master = np.concatenate((y_valid_master, y_valid_batch), axis=0)
            print("Updated master validation set. Shape =", X_valid_master.shape)
        model.fit(X_train_batch, y_train_batch, validation_data=(X_valid_master, y_valid_master), shuffle=True, nb_epoch=15, batch_size=1024)
    model.save('model.h5')


