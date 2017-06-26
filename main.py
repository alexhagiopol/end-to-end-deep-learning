import architecture
import utilities
import numpy as np

import pickle
import os
from sklearn.utils import shuffle

if __name__ == "__main__":
    # load data from pickle file
    data_file = 'pickle_data.p'
    utilities.batch_preprocess('data', ['ian_dataset_0', 'ian_dataset_1'], max_num_measurements=None)
    with open(os.path.join('data', data_file), mode='rb') as f:
        pickle_data = pickle.load(f)
    X = pickle_data['features']
    y = pickle_data['labels']

    print("features data shape", X.shape)
    print("labels data shape", y.shape)
    X, y = shuffle(X, y, random_state=0)
    X_train = X[0:int(0.9 * X.shape[0]), :, :]
    y_train = y[0:int(0.9 * y.shape[0])]
    X_test = X[int(0.9 * X.shape[0]):, :, :]
    y_test = y[int(0.9 * y.shape[0]):]
    print("train features data shape", X_train.shape)
    print("train labels data shape", y_train.shape)
    print("test features data shape", X_test.shape)
    print("test labels data shape", y_test.shape)
    model = architecture.nvidia_model_small()
    X_train = np.expand_dims(X_train, axis=3)
    '''
    print("X_train SHAPE BELOW:")
    print(X_train.shape)
    for i in range(X_train.shape[0]):
        utilities.show_image((1, 1, 1), 'image at index' + str(i), X_train[i, :, :, 0], width=3)
    '''
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, batch_size=1024)
    model.save('model.h5')
