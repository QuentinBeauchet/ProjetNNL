from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import cv2
import pandas as pd
from Boxes import Boxes
from PopUp import PopUp
from ModifyCategoriesPopUp import ModifyCategoriesPopUp
import os
from tkinter import messagebox


class ImageAnnotator:
    def __init__(self):
        self.root = Tk()
        self.root.title("ImageAnnotator")
        self.root.resizable(False, False)
        self.root.withdraw()
        # TODO créer les dossiers inputs/settings s'il n'existent pas
        self.imgPath = askopenfilename(
            initialdir="./inputs/", filetypes=[("Images", ".png .jpg")])
        if(len(self.imgPath) == 0):
            return
        self.imgName = os.path.basename(self.imgPath).split(".")[0]
        self.root.deiconify()
        self.loadImage()
        self.boxes = Boxes(self.canvas)
        data = pd.read_csv('settings/categories.csv')
        self.boxes.categories = data.iloc[:, 1].tolist()
        self.setMenu()
        self.setBinds()
        self.root.mainloop()

    def loadImage(self):
        self.img = ImageTk.PhotoImage(Image.open(self.imgPath))
        self.canvas = Canvas(
            self.root, width=self.img.width(), height=self.img.height())
        self.canvas.create_image(0, 0, anchor=NW, image=self.img)
        self.canvas.pack()
        self.root.eval('tk::PlaceWindow . center')

    def SaveAndloadImageFromMenu(self):
        self.save()
        self.root.destroy()
        ImageAnnotator()

    def setMenu(self):
        self.menu = Menu(self.root)
        # File
        self.menuFile = Menu(self.menu, tearoff=0)
        # Save All
        self.menuFile.add_command(
            label="Save All", accelerator="Ctrl+S", command=self.save)
        self.root.bind('<Control-s>', lambda _: self.save())
        # Load
        self.menuFile.add_command(
            label="Save & Load image", accelerator="Ctrl+L", command=self.SaveAndloadImageFromMenu)
        self.root.bind('<Control-l>', lambda _: self.loadImageFromMenu())
        # Save Boxes
        self.menuFile.add_command(
            label="Save Boxes", accelerator="Ctrl+B", command=self.writeData)
        self.root.bind('<Control-b>', lambda _: self.writeData())
        # Save Categories
        self.menuFile.add_command(
            label="Save Categories", accelerator="Ctrl+K", command=self.writeCategories)
        self.root.bind('<Control-k>', lambda _: self.writeCategories())
        # Import Boxes
        self.menuFile.add_command(
            label="Import Boxes", accelerator="Ctrl+I", command=self.importBoxes)
        self.root.bind('<Control-i>', lambda _: self.importBoxes())
        # Import Categories
        self.menuFile.add_command(
            label="Import Categories", accelerator="Ctrl+O", command=self.importCategories)
        self.root.bind('<Control-o>', lambda _: self.importCategories())

        self.menu.add_cascade(label="File", menu=self.menuFile)

        # Edit
        self.menuEdit = Menu(self.menu, tearoff=0)
        # Modifiy Categories
        self.menuEdit.add_command(
            label="Modify Categories", accelerator="Ctrl+M", command=self.popUpCategories)
        self.root.bind('<Control-m>', lambda _: self.popUpCategories())
        # Resolve conflicts
        self.menuEdit.add_command(
            label="Resolve conflicts", accelerator="Ctrl+R", command=self.boxes.clear)
        self.root.bind('<Control-r>', lambda _: self.boxes.clear())

        self.menu.add_cascade(label="Edit", menu=self.menuEdit)

        # Help
        self.menuHelp = Menu(self.menu, tearoff=0)
        # Controls
        self.menuHelp.add_command(
            label="Controls", accelerator="Ctrl+H", command=self.controls_help)
        self.root.bind('<Control-h>', lambda _: self.controls_help())
        # Shortcuts
        self.menuHelp.add_command(
            label="Shortcuts", accelerator="Ctrl+X", command=self.shortcuts_help)
        self.root.bind('<Control-x>', lambda _: self.shortcuts_help())
        # How conflicts work
        self.menuHelp.add_command(
            label="How conflicts work", accelerator="Ctrl+N", command=self.conflicts_help)
        self.root.bind('<Control-n>', lambda _: self.conflicts_help())
        # How to use categories
        self.menuHelp.add_command(
            label="How use categories", accelerator="Ctrl+P", command=self.categories_help)
        self.root.bind('<Control-p>', lambda _: self.categories_help())

        self.menu.add_cascade(label="Help", menu=self.menuHelp)

        self.root.config(menu=self.menu)

    def setBinds(self):
        self.root.bind('<Button-1>', self.boxes.click)
        self.root.bind('<ButtonRelease-1>', self.boxes.release)
        self.root.bind('<B1-Motion>', self.boxes.move)
        self.root.bind('<Button-3>', self.boxes.change)

    def popUpCategories(self):
        ModifyCategoriesPopUp(self.canvas, self.boxes)

    def save(self):
        self.saveImages()
        self.writeData()
        self.writeCategories()

    def saveImages(self):
        img = cv2.imread(self.imgPath)
        folderPath = os.path.join('outputs', self.imgName, "")
        if len(self.boxes.boxes) > 0:
            if not (os.path.exists(folderPath) and os.path.isdir(folderPath)):
                os.mkdir(folderPath)
        for i in range(len(self.boxes.boxes)):
            x1, y1, x2, y2 = self.boxes.boxes[i].coords()
            crop_img = img[int(min(y1, y2)):int(max(y1, y2)), int(
                min(x1, x2)):int(max(x1, x2))].copy()
            cv2.imwrite(folderPath + f"{i}.png", crop_img)

    def writeData(self):
        jsonArray = []
        for box in self.boxes.boxes:
            x1, y1, x2, y2 = box.coords()
            jsonArray.append({"x1": x1, "x2": x2, "y1": y1,
                             "y2": y2, "categories": box.categorie})
        df = pd.DataFrame(jsonArray)
        df.to_json(f'settings/{self.imgName}.json', orient='records')
        df.to_csv(f'settings/{self.imgName}.csv')

    def writeCategories(self):
        df = pd.DataFrame(self.boxes.categories, columns=["Categories"])
        df.to_json('settings/categories.json', orient='records')
        df.to_csv('settings/categories.csv')

    def importCategories(self):
        filePath = askopenfilename(initialdir="./settings/", initialfile="categories",
                                   filetypes=[("CSV or JSON Files", ".csv .json")])
        if(len(filePath) == 0):
            return
        if filePath.endswith(".csv"):
            data = pd.read_csv(filePath)
            self.boxes.categories = data.iloc[:, 1].tolist()
        else:
            data = pd.read_json(filePath)
            self.boxes.categories = data.iloc[:, 0].tolist()

    def importBoxes(self):
        filePath = askopenfilename(initialdir="./settings/", initialfile=f"{self.imgName}",
                                   filetypes=[("CSV or JSON Files", ".csv .json")])
        if(len(filePath) == 0):
            return
        if filePath.endswith(".csv"):
            data = pd.read_csv(filePath)
            for boxe in data.values.tolist():
                self.boxes.addBox(boxe[1], boxe[2], boxe[3], boxe[4], boxe[5])
        else:
            data = pd.read_json(filePath)
            for boxe in data.values.tolist():
                self.boxes.addBox(boxe[0], boxe[1], boxe[2], boxe[3], boxe[4])

    def controls_help(self):
        messagebox.showinfo(
            "Controls Help", "Left click to add a box\nRight click to select a box\n")

    def shortcuts_help(self):
        messagebox.showinfo("Shortcuts List", "Ctrl+S: Save All\nCtrl+L: Save & Load image\nCtrl+B: Save box\nCtrl+K: Save Categories\nCtrl+I: Import Boxes\nCtrl+O: Import Categories\nCtrl+M: Modify Categories\nCtrl+R: Resolve conflicts\nCtrl+H: Controls Help\nCtrl+X: Shortcuts Help\nCtrl+N: How conflicts work\nCtrl+P: How to use categories\n")

    def conflicts_help(self):
        messagebox.showinfo("How we resolve conflicts",
                            "If a box englobed an other box then we remove the box that is enclosed\nIf the intersection of two boxes take more than 20% of the area of the first box, the first box will be removed\n")

    def categories_help(self):
        messagebox.showinfo("How use categories", "You can add and delete categories with the menu (Ctrl+M)\nYou can import and save categories with the menu (Ctrl+I, Ctrl+K)\nTo apply a category to a box, just right click on the box and select your category at the list\nTo remove a category from a box , just right click on the box and select 'None' at the list\nYou can also apply categorie when you make the box\n")


ImageAnnotator()
