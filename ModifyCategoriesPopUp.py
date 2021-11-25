from tkinter import *

class ModifyCategoriesPopUp:
    def __init__(self, canvas, boxes):
        self.boxes = boxes
        self.popup = Toplevel(canvas, borderwidth=3, relief="ridge")
        self.fr = Frame(self.popup)
        self.fr.pack()
        self.my_text = Text(self.popup,width =20, height=1)
        self.my_text.pack()
        self.lbx = Listbox(self.fr, font = ("Verdana",16))
        self.categories = boxes.categories

    def setWindow(self):
        self.lbx.pack(side=LEFT, fill="both", expand=True)
        for i in range(len(self.categories)):
            self.lbx.insert(i, self.categories[i])
        sbr = Scrollbar(self.fr)
        sbr.pack(side=RIGHT, fill="y")
        sbr.config(command=self.lbx.yview)
        self.lbx.config(yscrollcommand=sbr.set)

    def addCategory(self):
        self.lbx.insert(0,self.my_text.get(1.0,END))
        self.my_text.delete(1.0,END)
        self.boxes.categories = self.lbx.get(0,END)

    def deleteCategory(self):
        self.lbx.delete(ANCHOR)
        self.boxes.categories = self.lbx.get(0, END)

    def setButton(self):
        buttonModify = Button(
            self.popup,
            text="Delete",
            font=("Verdana", 16),
            command=lambda:self.deleteCategory()
        )
        buttonModify.pack(side=LEFT)

        buttonAdd = Button(
            self.popup,
            text="Add",
            font=("Verdana", 16),
            command=lambda:self.addCategory()
        )
        buttonAdd.pack(side=LEFT)