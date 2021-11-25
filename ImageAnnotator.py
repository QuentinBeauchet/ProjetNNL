from tkinter import *
from PIL import ImageTk, Image  
import platform
import cv2
from shapely.geometry import box
from shapely.geometry import Point

root = Tk()
root.title("ImageAnnotator")
img = ImageTk.PhotoImage(Image.open("dessin.jpg"))  
canvas = Canvas(root, width = img.width(), height = img.height())  
canvas.pack()  
canvas.create_image(0, 0, anchor=NW, image=img)

class Box:
    def __init__(self):
        self.rect, self.polygon, self.clickEvent, self.categorie = None, None, None, None

    def __lt__(self, other):
        return self.polygon.area < other.polygon.area

    def click(self,event):
        self.clickEvent = event

    def update(self,event,outline='cyan'):
        self.polygon = box(self.clickEvent.x,self.clickEvent.y,event.x,event.y)
        self.draw(outline)

    def clear(self):
        if(self.rect is not None):
            canvas.delete(self.rect)

    def coords(self):
        x,y = self.polygon.exterior.coords.xy
        return (x[1],y[1],x[3],y[3])

    def size(self):
        x1,y1,x2,y2 = self.coords()
        return (abs(x2-x1),abs(y2-y1))

    def isValid(self):
        width, height = self.size()
        return self.polygon.area>=40 and width>=5 and height>=5

    def draw(self,outline):
        if(self.polygon is not None):
            self.clear()
            self.rect = canvas.create_rectangle(self.coords(), outline=(outline if self.isValid() else "red"), width=2)

class Boxes:
    def __init__(self):
        self.boxes = []
        self.currentBox = None

    def click(self,event):
        self.currentBox = Box()
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
        self.popup = PopUp()
        self.popup.setText(self.currentBox)
        self.popup.setOptionBar(default,["one", "two", "three"])
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

class PopUp:
    def __init__(self):
        self.canvas = Toplevel(root, borderwidth=3, relief="ridge")
        self.canvas.geometry("100x170")
        self.canvas.wm_attributes(*(['-topmost', True] if platform.system() == "Windows" else ['-type', 'splash','-topmost', True]))
        self.canvas.grid_rowconfigure(0,weight=1)
        self.canvas.grid_columnconfigure(0,weight=1)
        self.canvas.wait_visibility()
        self.canvas.grab_set()

    def setText(self,boxe):
        Label(self.canvas, text = "Dimensions:\nx1: {}\ny1: {}\nx2: {}\ny2: {}".format(*boxe.coords()), 
        font=('Arial 10 bold'), justify=LEFT).grid(row=0, column=0 , columnspan=2)

    def setOptionBar(self,default,categories):
        self.categorie = StringVar(self.canvas)
        self.categorie.set(default)
        options = OptionMenu(self.canvas, self.categorie, *categories)
        options.grid(row=1, column=0 , columnspan=2 , sticky="ew")

    def setButtons(self,boxes):
        Button(self.canvas, text ="OK", command = boxes.set).grid(row=2, column=0)
        Button(self.canvas, text ="DEL", command = boxes.delete).grid(row=2, column=1)

    def clear(self):
        self.canvas.destroy()
        self.canvas.grab_release()
            
boxes = Boxes()

root.bind('<Button-1>', boxes.click)
root.bind('<ButtonRelease-1>',boxes.release)
root.bind('<B1-Motion>', boxes.move)
root.bind('<Button-3>',boxes.change)
root.bind('<space>', boxes.clear)

root.mainloop() 