from tkinter import *

class popupWindow:
    def __init__(self,modifyCategoriesClass):
        top=self.top=Toplevel(modifyCategoriesClass.popup)
        self.modifyCategoriesClass = modifyCategoriesClass
        top.grab_set()
        self.l=Label(top,text="Modifiez la catégorie :")
        self.l.pack()
        self.hasConfirmed = False
        self.e = Entry(top)
        self.indexItemSelect = self.modifyCategoriesClass.lbx.curselection()[0]
        self.e.insert(END, self.modifyCategoriesClass.lbx.get(self.indexItemSelect))
        self.e.bind('<Return>', lambda event : self.cleanup())
        self.e.pack()
        self.b=Button(top,text='CONFIRMER',command=self.cleanup)
        self.b.pack()

    def cleanup(self):
        self.modifyCategoriesClass.lbx.delete(self.indexItemSelect)
        self.modifyCategoriesClass.lbx.insert(self.indexItemSelect,self.e.get())
        self.hasConfirmed = True
        self.top.destroy()

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
            command=lambda: self.popupModify()
        )
        self.buttonModify.pack(side=LEFT)
        self.buttonModify["state"] = "disabled"

    def popupModify(self):
        if len(self.lbx.curselection()) == 1 :
            self.w = popupWindow(self)
            self.buttonModify["state"] = "disabled"
            self.popup.wait_window(self.w.top)
            if not(self.w.hasConfirmed) :
                self.buttonModify["state"] = "normal"

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

    def _update_listbox(self):
        self.lb.delete(1)
        self.lb.insert(1, time.asctime())

    def onselect(self,evt):
        w = evt.widget
        nbItemSelected = len(w.curselection())
        if nbItemSelected == 1:
            self.buttonModify["state"] = "normal"
        else :
            self.buttonModify["state"] = "disabled"