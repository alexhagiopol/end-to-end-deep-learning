from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Conv2D, ELU, Dropout
from keras.utils import plot_model


def test_model_1(input_shape=(67, 320)):
    # create model
    model = Sequential()
    # normalize pixels to between -1 <= x <= 1
    model.add(Lambda(lambda x: (x / 255.0) * 2 - 1, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model


def test_model_2():
    model = Sequential()
    model.add(Flatten(input_shape=(67, 320)))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model
