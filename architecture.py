from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Conv2D, ELU, Dropout
from keras.utils import plot_model


def test_model():
    model = Sequential()
    model.add(Flatten(input_shape=(67, 320)))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model


def nvidia_model():
    """
    See https://arxiv.org/pdf/1604.07316.pdf. Designed for 3 layer RGB input.
    """
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=(67, 320, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model

def nvidia_model_small():
    """
    Designed for single layer grayscale input.
    """
    model = Sequential()
    model.add(Conv2D(8, (5, 5), strides=(2, 2), activation="relu", input_shape=(67, 320, 1)))
    model.add(Conv2D(12, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(16, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(24, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Conv2D(24, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    return model
