import architecture
import utilities
import sys


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Incorrect syntax.")
        print("Example syntax: python main.py udacity_dataset model.h5 3000")
        print("Randomizing the order of your dataset is recommended especially when training for the first time after recording the dataset.")
        print("Optional syntax: python main.py udacity_dataset model.h5 3000 randomize")
        sys.exit()
    # get arguments from user
    image_input_dir = sys.argv[1]
    model_path = sys.argv[2]
    batch_size = int(sys.argv[3])
    if len(sys.argv) == 5 and sys.argv[4] == 'randomize':
        dataset_log_path = utilities.get_driving_log_path(image_input_dir)
        print("Randomizing dataset at", dataset_log_path)
        utilities.randomize_dataset_csv(dataset_log_path)
    measurement_index = 0  # index of measurements in dataset
    dataset_log = utilities.get_dataset_from_csv(image_input_dir)
    dataset_size = dataset_log.shape[0]
    # use first 20% of dataset for validation
    validation_batch_size = int(0.2 * dataset_size)
    validation_set = utilities.batch_preprocess(image_input_dir, measurement_range=(measurement_index, validation_batch_size))
    X_valid = validation_set['features']
    y_valid = validation_set['labels']
    measurement_index = validation_batch_size  # update measurement index to the end of the validation set
    model = architecture.nvidia_model()  # initialize neural network model that will be iteratively trained in batches
    while measurement_index < dataset_size:
        end_index = measurement_index + batch_size
        if end_index < dataset_size:
            print("Pre-processing from index", measurement_index, "to index", end_index)
            preprocessed_batch = utilities.batch_preprocess(image_input_dir, measurement_range=(measurement_index, end_index))
        else:
            print("Pre-processing from index", measurement_index, "to index", dataset_size)
            preprocessed_batch = utilities.batch_preprocess(image_input_dir, measurement_range=(measurement_index, None))
        X_batch = preprocessed_batch['features']
        y_batch = preprocessed_batch['labels']
        print("Done preprocessing.")
        print("features data shape", X_batch.shape)
        print("labels data shape", y_batch.shape)
        model.fit(X_batch, y_batch, validation_data=(X_valid, y_valid), shuffle=True, nb_epoch=15, batch_size=1024)
        measurement_index += batch_size
    model.save(model_path)
