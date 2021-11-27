from tkinter import *
from PIL import ImageTk, Image
import cv2
import pandas as pd
from Boxes import Boxes
from ModifyCategoriesPopUp import ModifyCategoriesPopUp

root = Tk()
root.title("ImageAnnotator")
img = ImageTk.PhotoImage(Image.open("dessin.jpg"))
canvas = Canvas(root, width = img.width(), height = img.height())
canvas.pack()
canvas.create_image(0, 0, anchor=NW, image=img)

def writeFiles(boxes):
    jsonArray = []
    for box in boxes:
        x1,y1,x2,y2 = box.coords()
        jsonArray.append({"x1" : x1, "x2" : x2, "y1" : y1, "y2" : y2, "categories" : box.categorie})
    df = pd.DataFrame(jsonArray)
    df.to_json('data.json', orient = 'records')
    df.to_csv('data.csv', index_label = 'index')

def save(boxes):
    img = cv2.imread("dessin.jpg")
    for i in range(len(boxes.boxes)):
        x1,y1,x2,y2 = boxes.boxes[i].coords()
        crop_img = img[int(y1):int(y2), int(x1):int(x2)].copy()
        cv2.imwrite("save/" + str(i) + ".png", crop_img)
    writeFiles(boxes.boxes)
          
boxes = Boxes(canvas)
my_menu = Menu(root)

file_menu = Menu(my_menu, tearoff=0)
file_menu.add_command(label="Save", command = lambda:save(boxes))
file_menu.add_command(label="Import")
my_menu.add_cascade(label="File",menu=file_menu)

def popUpCategories():
    ModifyCategoriesPopUp(canvas, boxes)

edit_menu = Menu(my_menu, tearoff=0)
edit_menu.add_command(label="Modify Categories",command=lambda:popUpCategories())
edit_menu.add_command(label="Import Categories")
my_menu.add_cascade(label="Edit",menu=edit_menu)
root.config(menu=my_menu)

root.bind('<Button-1>', boxes.click)
root.bind('<ButtonRelease-1>',boxes.release)
root.bind('<B1-Motion>', boxes.move)
root.bind('<Button-3>',boxes.change)
root.bind('<space>', boxes.clear)

root.mainloop() 