import utilities
import pickle
import os
from sklearn.utils import shuffle

if __name__ == "__main__":
    # load data from pickle file
    data_file = 'pickle_data.p'
    utilities.preprocess('data', 'images', max_num_measurements=100)
    with open(os.path.join('data', data_file), mode='rb') as f:
        pickle_data = pickle.load(f)
    X = pickle_data['features']
    y = pickle_data['labels']

    print("features data shape", X.shape)
    print("labels data shape", y.shape)
    X, y = shuffle(X, y, random_state=0)
    X_train = X[0:int(0.8 * X.shape[0]), :, :]
    y_train = y[0:int(0.8 * y.shape[0])]
    X_valid = X[int(0.8 * X.shape[0]):int(0.9 * X.shape[0]), :, :]
    y_valid = y[int(0.8 * y.shape[0]):int(0.9 * y.shape[0])]
    X_test = X[int(0.9 * X.shape[0]):, :, :]
    y_test = y[int(0.9 * y.shape[0]):]
    print("train features data shape", X_train.shape)
    print("train labels data shape", y_train.shape)
    print("valid features data shape", X_valid.shape)
    print("valid labels data shape", y_valid.shape)
    print("test features data shape", X_test.shape)
    print("test labels data shape", y_test.shape)
    '''
    for i in range(X_train.shape[0]):
        utilities.show_image((1, 1, 1), 'image at index' + str(i), X[i, :, :], width=3)
    '''
