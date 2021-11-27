from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import cv2
import pandas as pd
from Boxes import Boxes
from ModifyCategoriesPopUp import ModifyCategoriesPopUp

class ImageAnnotator:
    def __init__(self):
        self.root = Tk()
        self.root.title("ImageAnnotator")
        self.root.resizable(False, False)

        self.root.withdraw()
        self.imgPath = askopenfilename(filetypes=[("Images", ".png .jpg")])
        if(self.imgPath == ()):
            return
        self.root.deiconify()

        self.loadImage()
        self.boxes = Boxes(self.canvas)
        self.setMenu()
        self.setBinds()

        self.root.mainloop() 
    
    def loadImage(self):
        self.img = ImageTk.PhotoImage(Image.open(self.imgPath))
        self.canvas = Canvas(self.root, width = self.img.width(), height = self.img.height())
        self.canvas.create_image(0, 0, anchor=NW, image=self.img)
        self.canvas.pack()
        self.root.eval('tk::PlaceWindow . center')

    def setMenu(self):
        self.menu = Menu(self.root)

        self.menuFile = Menu(self.menu, tearoff=0)
        self.menuFile.add_command(label="Save", command = self.save)
        self.menuFile.add_command(label="Import")
        self.menu.add_cascade(label="File",menu=self.menuFile)

        self.menuEdit = Menu(self.menu, tearoff=0)
        self.menuEdit.add_command(label="Modify Categories", command = self.popUpCategories)
        self.menuEdit.add_command(label="Import Categories")
        self.menuEdit.add_command(label="Clear Boxes", command = self.boxes.clear)
        self.menu.add_cascade(label="Edit",menu=self.menuEdit)

        self.root.config(menu=self.menu)

    def setBinds(self):
        self.root.bind('<Button-1>', self.boxes.click)
        self.root.bind('<ButtonRelease-1>',self.boxes.release)
        self.root.bind('<B1-Motion>', self.boxes.move)
        self.root.bind('<Button-3>',self.boxes.change)
        self.root.bind('<space>', self.boxes.clear)

    def popUpCategories(self):
        ModifyCategoriesPopUp(self.canvas, self.boxes)

    def save(self):
        img = cv2.imread(self.imgPath)
        for i in range(len(self.boxes.boxes)):
            x1,y1,x2,y2 = self.boxes.boxes[i].coords()
            crop_img = img[int(min(y1,y2)):int(max(y1,y2)), int(min(x1,x2)):int(max(x1,x2))].copy()
            cv2.imwrite("save/{}.png".format(i), crop_img)
        self.writeFiles()

    def writeFiles(self):
        jsonArray = []
        for box in self.boxes.boxes:
            x1,y1,x2,y2 = box.coords()
            jsonArray.append({"x1" : x1, "x2" : x2, "y1" : y1, "y2" : y2, "categories" : box.categorie})
        df = pd.DataFrame(jsonArray)
        df.to_json('data.json', orient = 'records')
        df.to_csv('data.csv')

ImageAnnotator()