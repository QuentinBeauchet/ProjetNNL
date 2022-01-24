from Models import ClassificationModel
from Models import LocalisationModel
import matplotlib.pyplot as plt
import cv2


def evaluate(CustomModel):
    score = CustomModel.model.evaluate(
        CustomModel.X_test, CustomModel.y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


def predict(imgPath, mode="probabilities"):
    Lmodel = LocalisationModel()
    Lmodel.loadModel()
    Lmodel_SIZE = Lmodel.model.layers[0].get_output_at(0).get_shape()[1]

    Cmodel = ClassificationModel()
    Cmodel.loadModel()
    Cmodel_SIZE = Cmodel.model.layers[0].get_output_at(0).get_shape()[1]

    # Chargement de l'image
    img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(imgGray, (Lmodel_SIZE, Lmodel_SIZE))

    # Prediction de la bouding box
    bbox = Lmodel.predict(res)
    rw, rh = (imgGray.shape[1]/Lmodel_SIZE), (imgGray.shape[0]/Lmodel_SIZE)

    # Transformation de la bounding box predite pour l'image
    xmin, ymin, xmax, ymax = int(
        bbox[0]*rw), int(bbox[1]*rh), int(bbox[2]*rw), int(bbox[3]*rh)

    rect = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

    # Prediction de la categorie
    categorie = Cmodel.predict(cv2.resize(
        imgGray[ymin:ymax, xmin:xmax], (Cmodel_SIZE, Cmodel_SIZE)))

    # Affiche le resultat
    plt.title(f"Mask :{categorie[1][0][0]:.2f}% | No Mask: {categorie[1][0][1]:.2f}%" if mode ==
              "probabilities" else f"Categorie: {categorie[0]}")
    plt.imshow(rect)
    plt.show()
