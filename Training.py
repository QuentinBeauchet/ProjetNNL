import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

PATH = "outputs"
CATEGORIES = ["Mask", "Nude"]
IMG_SIZE = 50


def createTrainingData():
    X = []
    y = []
    for category in CATEGORIES:
        path = os.path.join(PATH, category)
        classIndex = CATEGORIES.index(category)
        for img in os.listdir(path):
            imgArray = cv2.imread(os.path.join(
                path, img), cv2.IMREAD_GRAYSCALE)
            newArray = cv2.resize(imgArray, (IMG_SIZE, IMG_SIZE))
            X.append(newArray)
            y.append(classIndex)
    return np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1), np.array(y)


def saveData(X, y):
    pickleOut = open("X.pickle", "wb")
    pickle.dump(X, pickleOut)
    pickleOut.close()

    pickleOut = open("y.pickle", "wb")
    pickle.dump(y, pickleOut)
    pickleOut.close()


def loadData():
    X = pickle.load(open("X.pickle", "rb"))
    y = pickle.load(open("y.pickle", "rb"))
    return X, y


if __name__ == "__main__":
    X, y = createTrainingData()
    #saveData(X, y)
    #X = pickle.load(open("X.pickle", "rb"))
    #y = pickle.load(open("y.pickle", "rb"))
    X = X/255.0

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3)
