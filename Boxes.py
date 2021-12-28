from PopUp import PopUp
from Box import Box
from shapely.geometry import Point
from shapely.geometry import box


class Boxes:
    def __init__(self, canvas):
        self.boxes = []
        self.currentBox = None
        self.categories = ["None"]
        self.canvas = canvas

    def click(self, event):
        self.currentBox = Box(self.canvas)
        self.currentBox.click(event)

    def release(self, event):
        if self.currentBox != None :
            self.currentBox.update(event, 'black')
        if(self.currentBox.isValid()):
            self.boxes.append(self.currentBox)
            self.popUp()
        else:
            self.currentBox.clear()
            self.currentBox = None

    def addBox(self, x1, x2, y1, y2, categorie):
        boxe = Box(self.canvas)
        boxe.polygon = box(x1, y1, x2, y2)
        boxe.categorie = categorie
        self.boxes.append(boxe)
        boxe.draw('black')

    def move(self, event):
        if(self.currentBox is not None):
            self.currentBox.update(event)

    def popUp(self, default="None"):
        self.popup = PopUp(self.canvas)
        self.popup.setText(self.currentBox)
        self.popup.setOptionBar(default, self.categories)
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

    def change(self, event):
        point = Point(event.x, event.y)
        for boxe in self.boxes:
            if boxe.polygon.contains(point):
                self.currentBox = boxe
                self.currentBox.draw("cyan")
                self.popUp(self.currentBox.categorie)
                break

    def draw(self):
        for box in self.boxes:
            box.draw()

    def notifyCategoryDeletion(self, category):
        for boxe in self.boxes:
            if(boxe.categorie == category):
                boxe.categorie = None

    def notifyCategoryModification(self, oldCategory, newCategory):
        for boxe in self.boxes:
            if (boxe.categorie == oldCategory):
                boxe.categorie = newCategory

    def clear(self):
        self.boxes.sort(reverse=True)
        for box1 in self.boxes:
            total = None
            for box2 in self.boxes:
                if(box1 != box2):
                    intersection = box1.polygon.intersection(box2.polygon)
                    if(box2.polygon.area == intersection.area):
                        box1.clear()
                        self.boxes.remove(box1)
                        break
                    total = intersection if (
                        total is None) else total.union(intersection)
            if(total is not None):
                percent = (total.area*100)/box1.polygon.area
                if(20 < percent < 100):
                    box1.clear()
                    self.boxes.remove(box1)
