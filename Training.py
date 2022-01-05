import math
import pickle
import cv2
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras import models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from collections import Counter
from sklearn.utils import shuffle

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

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
    return shuffle(np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1), np.array(y), random_state=42)


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


def showLayers(classifier, img):
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    plt.imshow(img_tensor[0])
    plt.show()

    # Extracts the outputs of the top 12 layers
    layer_outputs = [layer.output for layer in classifier.layers[:12]]
    # Creates a model that will return these outputs, given the model input
    activation_model = models.Model(
        inputs=classifier.input, outputs=layer_outputs)
    # Returns a list of five Numpy arrays: one array per layer activation
    activations = activation_model.predict(img_tensor)

    fig = plt.figure(1)
    index = 0
    k = 1
    for layer in activations:
        ax = fig.add_subplot(2, 5, k)
        channel_image = layer
        # Post-processes the feature to make it visually palatable
        try:
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            channel_image = [[[np.mean(j)] for j in i] for i in layer[0]]
        except:
            pass
        ax.imshow(
            channel_image, aspect='auto', cmap='viridis')
        ax.axis('off')
        ax.set_title(classifier.layers[index].name)
        k += 1
        index += 1

    plt.show()


if __name__ == "__main__":
    X, y = createTrainingData()
    print(Counter(y))
    img = X[1]
    # saveData(X, y)
    # X = pickle.load(open("X.pickle", "rb"))
    # y = pickle.load(open("y.pickle", "rb"))
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

    model.fit(X, y, batch_size=16, epochs=10, validation_split=0.1)

    img_path = 'outputs/Mask/maksssksksss53-bb-323x53-45-54.png'
    img = load_img(img_path, target_size=(
        IMG_SIZE, IMG_SIZE), color_mode="grayscale")

    # predicting images
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    predict = model.predict(images, batch_size=10)
    classes = np.argmax(predict, axis=1)
    print("Predicted class is:", [CATEGORIES[i] for i in classes])
