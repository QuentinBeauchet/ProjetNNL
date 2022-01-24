from venv import create
from xml.dom import minidom
import os
import cv2
import numpy as np
import pickle
from numpy import extract
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, LeakyReLU, BatchNormalization, MaxPool2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


Categories = ["Mask", "Nude"]


def saveData(data, name):
    pickleOut = open(name + ".pickle", "wb")
    pickle.dump(data, pickleOut)
    pickleOut.close()


def evaluate(CustomModel):
    score = CustomModel.model.evaluate(
        CustomModel.X_test, CustomModel.y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


class ClassificationModel:
    def __init__(self, IMG_SIZE=120):
        self.IMG_SIZE = IMG_SIZE

    def extractData(self, validation_split=0.1):
        Mask, Nude = [], []
        for file in os.listdir("annotations"):
            XML = os.path.join("annotations", file)
            xmldoc = minidom.parse(XML)
            itemlist = xmldoc.getElementsByTagName('object')
            img = cv2.imread(os.path.join(
                "images", os.path.splitext(file)[0]) + ".png", cv2.IMREAD_GRAYSCALE)

            for object in itemlist:
                categorie = object.childNodes[1].firstChild.nodeValue
                bbox = object.childNodes[11]
                xmin, ymin, xmax, ymax = (int(bbox.childNodes[1].firstChild.nodeValue),
                                          int(bbox.childNodes[3].firstChild.nodeValue),
                                          int(bbox.childNodes[5].firstChild.nodeValue),
                                          int(bbox.childNodes[7].firstChild.nodeValue))
                crop = cv2.resize(img[ymin:ymax, xmin:xmax],
                                  (self.IMG_SIZE, self.IMG_SIZE))
                if(categorie == "without_mask"):
                    Nude.append(crop)
                else:
                    Mask.append(crop)
        np.random.shuffle(Mask)
        Mask = Mask[:(len(Nude))]

        X = Mask + Nude
        y = [[1, 0] for x in Mask] + [[0, 1] for x in Nude]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.array(
            X).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1), np.array(y), test_size=validation_split, random_state=42)
        saveData(self.X_train, "X_train_Classification")
        saveData(self.y_train, "y_train_Classification")
        saveData(self.X_test, "X_test_Classification")
        saveData(self.y_test, "y_test_Classification")
        print("Classification data saved !")

    def loadData(self):
        try:
            self.X_train = pickle.load(
                open("X_train_Classification.pickle", "rb"))
            self.y_train = pickle.load(
                open("y_train_Classification.pickle", "rb"))
            self.X_test = pickle.load(
                open("X_test_Classification.pickle", "rb"))
            self.y_test = pickle.load(
                open("y_test_Classification.pickle", "rb"))
        except:
            print("Data files not founds")

    def loadModel(self):
        try:
            self.model = load_model(
                'Models/ClassificationModel.h5')
        except:
            print("Model file not found")

    def train(self, batch_size=32, epochs=20):
        self.model = Sequential()

        self.model.add(Rescaling(1./255))
        self.model.add(RandomFlip())
        self.model.add(RandomRotation(0.2))

        self.model.add(Conv2D(8, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(16, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(128))

        self.model.add(Dense(2))
        self.model.add(Activation("softmax"))

        self.model.compile(loss="categorical_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

        self.model.fit(self.X_train, self.y_train, batch_size=batch_size,
                       epochs=epochs, validation_data=(self.X_test, self.y_test))

        self.model.save("Models/ClassificationModel.h5", save_format='h5')

    def predict(self, crop):
        predictions = self.model.predict(np.array([crop, ]))
        return (Categories[np.argmax(predictions, axis=1)[0]], predictions)


class LocalisationModel:
    def __init__(self, IMG_SIZE=120):
        self.IMG_SIZE = IMG_SIZE

    def extractData(self, validation_split=0.1):
        X, y = [], []
        for file in os.listdir("annotations"):
            XML = os.path.join("annotations", file)
            xmldoc = minidom.parse(XML)
            itemlist = xmldoc.getElementsByTagName('object')
            if(len(itemlist) == 1):
                img = cv2.imread(os.path.join(
                    "images", os.path.splitext(file)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
                crop = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                bbox = itemlist[0].childNodes[11]
                xmin, ymin, xmax, ymax = (int(bbox.childNodes[1].firstChild.nodeValue), int(bbox.childNodes[3].firstChild.nodeValue), int(
                    bbox.childNodes[5].firstChild.nodeValue), int(bbox.childNodes[7].firstChild.nodeValue))
                if(xmax-xmin > 100 and ymax-ymin > 100):
                    rw, rh = (
                        img.shape[1]/self.IMG_SIZE), (img.shape[0]/self.IMG_SIZE)
                    X.append(crop)
                    y.append([int(xmin/rw),
                              int(ymin/rh),
                              int(xmax/rw),
                              int(ymax/rh)])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(np.array(
            X).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1), np.array(y), test_size=validation_split, random_state=42)
        saveData(self.X_train, "X_train_Localisation")
        saveData(self.y_train, "y_train_Localisation")
        saveData(self.X_test, "X_test_Localisation")
        saveData(self.y_test, "y_test_Localisation")
        print(f"X: {len(X)}   y: {len(y)}")
        print("Localisation data saved !")

    def loadData(self):
        try:
            self.X_train = pickle.load(
                open("X_train_Localisation.pickle", "rb"))
            self.y_train = pickle.load(
                open("y_train_Localisation.pickle", "rb"))
            self.X_test = pickle.load(open("X_test_Localisation.pickle", "rb"))
            self.y_test = pickle.load(open("y_test_Localisation.pickle", "rb"))
            return
        except OSError:
            print("Data files not founds", "\n", "Generating them...")
            self.extractData()

    def loadModel(self):
        try:
            self.model = load_model(
                'Models/LocalisationModel.h5')
        except OSError:
            print("Model file not found", "\n", "Generating it...")
            self.createModel()

    def createModel(self):
        self.model = Sequential()

        self.model.add(Rescaling(1./255))

        # Input dimensions: (None, 96, 96, 3)
        self.model.add(Conv2D(32, (3, 3), padding='same',
                              use_bias=False, input_shape=(self.IMG_SIZE, self.IMG_SIZE, 1)))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        # Input dimensions: (None, 96, 96, 32)
        self.model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        # Input dimensions: (None, 48, 48, 32)
        self.model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        # Input dimensions: (None, 48, 48, 64)
        self.model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        # Input dimensions: (None, 24, 24, 64)
        self.model.add(Conv2D(96, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        # Input dimensions: (None, 24, 24, 96)
        self.model.add(Conv2D(96, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        # Input dimensions: (None, 12, 12, 96)
        self.model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        # Input dimensions: (None, 12, 12, 128)
        self.model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        # Input dimensions: (None, 6, 6, 128)
        self.model.add(Conv2D(256, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        # Input dimensions: (None, 6, 6, 256)
        self.model.add(Conv2D(256, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        # Input dimensions: (None, 3, 3, 256)
        self.model.add(Conv2D(512, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        # Input dimensions: (None, 3, 3, 512)
        self.model.add(Conv2D(512, (3, 3), padding='same', use_bias=False))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        # Input dimensions: (None, 3, 3, 512)
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='linear'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(4))

        self.model.compile(optimizer='adam', loss='mean_squared_error',
                           metrics=['accuracy'])

    def train(self, batch_size=32, epochs=20):
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=10, verbose=1)
        stop = EarlyStopping(monitor="val_loss", min_delta=0.1,
                             patience=40, restore_best_weights=True)

        self.model.fit(self.X_train, self.y_train, batch_size=batch_size,
                       epochs=epochs, validation_data=(self.X_test, self.y_test), callbacks=[reduce_lr, stop])

        self.model.save("Models/LocalisationModel.h5", save_format='h5')

    def predict(self, crop):
        predictions = self.model.predict(np.array([crop, ]))
        return [int(x) for x in predictions[0]]
