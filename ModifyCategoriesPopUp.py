from tkinter import *

class ModifyCategoriesPopUp:
    def __init__(self, canvas, boxes):
        self.boxes = boxes
        self.popup = Toplevel(canvas, borderwidth=3, relief="ridge")
        self.fr = Frame(self.popup)
        self.fr.pack()
        self.my_text = Text(self.popup,width =20, height=1)
        self.my_text.pack()
        self.lbx = Listbox(self.fr, selectmode = EXTENDED,font = ("Verdana",16))
        self.categories = boxes.categories

        self.my_text.bind('<Return>', lambda x=None:self.addCategory())

        self.lbx.pack(side=LEFT, fill="both", expand=True)
        for i in range(len(self.categories)):
            self.lbx.insert(i, self.categories[i])
        sbr = Scrollbar(self.fr)
        sbr.pack(side=RIGHT, fill="y")
        sbr.config(command=self.lbx.yview)
        self.lbx.config(yscrollcommand=sbr.set)

        buttonModify = Button(
            self.popup,
            text="Delete",
            font=("Verdana", 16),
            command=lambda: self.deleteCategory()
        )
        buttonModify.pack(side=LEFT)

        buttonAdd = Button(
            self.popup,
            text="Add",
            font=("Verdana", 16),
            command=lambda: self.addCategory()
        )
        buttonAdd.pack(side=LEFT)

    def addCategory(self):
        if not(str(self.my_text.get(1.0,END)).isspace()):
            self.lbx.insert(0,self.my_text.get(1.0,END))
            self.boxes.categories = self.lbx.get(0, END)
        self.my_text.delete(1.0,END)

    def deleteCategory(self):
        sel = self.lbx.curselection()
        for index in sel[::-1]:
            self.lbx.delete(index)
        self.boxes.categories = self.lbx.get(0, END)
