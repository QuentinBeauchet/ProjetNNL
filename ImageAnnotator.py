from tkinter import *
from PIL import ImageTk, Image
import json
import csv
from Boxes import Boxes
from ModifyCategoriesPopUp import ModifyCategoriesPopUp

root = Tk()
root.title("ImageAnnotator")
img = ImageTk.PhotoImage(Image.open("dessin.jpg"))
canvas = Canvas(root, width = img.width(), height = img.height())
canvas.pack()
canvas.create_image(0, 0, anchor=NW, image=img)

def makeJson(boxes):
    json_file = dict()
    json_file['box'] = []
    for i in range(len(boxes)):
        box = boxes[i]
        x1,y1,x2,y2 = box.coords()
        json_file['box'].append({"index" : i, "x1" : x1, "x2" : x2, "y1" : y1, "y2" : y2, "categories" : box.categorie})
    with open('data.json', 'w') as outfile:
        json.dump(json_file, outfile)
    jsonToCsv(json_file)

def jsonToCsv(json):
    data_file = open('data_file.csv', 'w', newline='')
    csv_writer = csv.writer(data_file)
    count = 0
    for emp in json['box']:
        if count == 0:
            header = emp.keys()
            csv_writer.writerow(header)
            count += 1
        csv_writer.writerow(emp.values())
    data_file.close()
          
boxes = Boxes(canvas)
my_menu = Menu(root)

file_menu = Menu(my_menu)
file_menu.add_command(label="Save", command= lambda: boxes.save())
file_menu.add_command(label="Import")
my_menu.add_cascade(label="File",menu=file_menu)

def popUpCategories():
    ModifyCategoriesPopUp(canvas, boxes)

edit_menu = Menu(my_menu)
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