from Models import ClassificationModel, evaluate, predict
from Models import LocalisationModel
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    predict("images/maksssksksss17.png", mode="categories")
    """
    Cmodel = ClassificationModel()
    Cmodel.loadData()
    Cmodel.loadModel()
    # Cmodel.train(epochs=200)
    evaluate(Cmodel)

    Lmodel = LocalisationModel()
    # Lmodel.extractData()
    Lmodel.loadData()
    Lmodel.loadModel()
    # Lmodel.train(epochs=500)
    # Lmodel.loadModel()
    evaluate(Lmodel)

    for i in range(len(Lmodel.X_test)):
        img = Lmodel.X_test[i]
        bbox = Lmodel.y_test[i]
        predictions = Lmodel.predict(img)

        rectPred = cv2.rectangle(
            img, (predictions[0], predictions[1]), (predictions[2], predictions[3]), (255, 0, 0), 1)
        rectBbox = cv2.rectangle(
            rectPred, (bbox[0], bbox[1], bbox[2], bbox[3]), (0, 255, 0), 1)

        plt.imshow(rectBbox)
        plt.show()

        crop = cv2.resize(img[predictions[1]:predictions[3],
                              predictions[0]:predictions[2]], (120, 120))

        print(Cmodel.predict(crop))
    """
