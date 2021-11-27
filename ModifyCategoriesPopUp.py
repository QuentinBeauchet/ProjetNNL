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
        self.lbx.bind('<<ListboxSelect>>', self.onselect)
        self.categories = boxes.categories

        self.my_text.bind('<Return>', lambda x=None:self.addCategory())

        self.lbx.pack(side=LEFT, fill="both", expand=True)
        for i in range(len(self.categories)):
            self.lbx.insert(i, self.categories[i])
        sbr = Scrollbar(self.fr)
        sbr.pack(side=RIGHT, fill="y")
        sbr.config(command=self.lbx.yview)
        self.lbx.config(yscrollcommand=sbr.set)

        self.buttonDelete = Button(
            self.popup,
            text="Delete",
            font=("Verdana", 16),
            command=lambda: self.deleteCategory()
        )
        self.buttonDelete.pack(side=LEFT)
        self.buttonDelete["state"] = "disabled"

        self.buttonAdd = Button(
            self.popup,
            text="Add",
            font=("Verdana", 16),
            command=lambda: self.addCategory()
        )
        self.buttonAdd.pack(side=LEFT)

        self.buttonModify = Button(
            self.popup,
            text="Modify",
            font=("Verdana", 16),
            command=lambda: self.modifyCategory()
        )
        self.buttonModify.pack(side=LEFT)
        self.buttonModify["state"] = "disabled"

    def modifyCategory(self):
        index = self.lbx.curselection()
        text = self.my_text.get(1.0,END).strip()
        if text != "" and index != ():
            self.lbx.delete(index)
            self.lbx.insert(index,text)
            self.lbx.select_set(index)

    def addCategory(self):
        text = self.my_text.get(1.0,END).strip()
        if text != "":
            self.lbx.insert(END,text)
            self.boxes.categories.append(text)

    def deleteCategory(self):
        index = self.lbx.curselection()
        if index != ():
            self.boxes.categories.remove(self.lbx.get(index))
            self.lbx.delete(index)

    def popupModify(self):
        if len(self.lbx.curselection()) == 1 :
            self.w = popupWindow(self)
            self.buttonModify["state"] = "disabled"
            self.popup.wait_window(self.w.top)
            if not(self.w.hasConfirmed) :
                self.buttonModify["state"] = "normal"

    def _update_listbox(self):
        self.lb.delete(1)
        self.lb.insert(1, time.asctime())

    def onselect(self,evt):
        w = evt.widget
        nbItemSelected = len(w.curselection())
        if nbItemSelected == 1:
            self.buttonModify["state"] = "normal"
            self.buttonDelete["state"] = "normal"
        else :
            self.buttonModify["state"] = "disabled"
            self.buttonDelete["state"] = "disabled"