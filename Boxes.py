from PopUp import PopUp
from Box import Box
from shapely.geometry import Point
from tkinter import Button
import cv2

class Boxes:
    def __init__(self,canvas):
        self.boxes = []
        self.currentBox = None
        self.categories = ["one", "two", "three"]
        self.canvas = canvas

    def click(self,event):
        self.currentBox = Box(self.canvas)
        self.currentBox.click(event)

    def release(self,event):
        self.currentBox.update(event,'black')
        if(self.currentBox.isValid()):
            self.boxes.append(self.currentBox)
            self.popUp()
        else:
            self.currentBox.clear()
            self.currentBox = None

    def move(self,event):
        if(self.currentBox is not None):
            self.currentBox.update(event)

    def popUp(self,default="one"):
        self.popup = PopUp(self.canvas)
        self.popup.setText(self.currentBox)
        self.popup.setOptionBar(default,self.categories)
        self.popup.setButtons(self)

    def set(self):
        self.popup.clear()
        self.currentBox.draw("black")
        self.currentBox.categorie = self.popup.categorie.get()

    def delete(self):
        self.popup.clear()
        self.currentBox.clear()
        self.boxes.remove(self.currentBox)
        self.currentBox = None

    def change(self,event):
        point = Point(event.x,event.y)
        for boxe in self.boxes:
            if boxe.polygon.contains(point):
                self.currentBox = boxe
                self.currentBox.draw("cyan")
                self.popUp(self.currentBox.categorie)
                break

    def draw(self):
        for box in self.boxes:
            box.draw()

    def clear(self,event):
        self.boxes.sort(reverse=True)
        for box1 in self.boxes:
            total = None
            for box2 in self.boxes:
                if(box1!=box2):
                    intersection = box1.polygon.intersection(box2.polygon)
                    if(box2.polygon.area==intersection.area):
                        box1.clear()
                        self.boxes.remove(box1)
                        break
                    total = intersection if (total is None) else total.union(intersection)
            if(total is not None):
                percent = (total.area*100)/box1.polygon.area
                if(20 < percent < 100):
                    box1.clear()
                    self.boxes.remove(box1)

    def save(self):
        img = cv2.imread("dessin.jpg")
        crop_img = img[self.currentBox.y1:self.currentBox.y2, self.currentBox.x1:self.currentBox.x2].copy()
        cv2.imwrite("save/" + str(self.currentBox.x1) + ".png", crop_img)
        self.makeJson()
        print('Successfully saved')