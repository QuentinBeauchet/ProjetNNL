from tkinter import *


class ModifyCategoriesPopUp:
    def __init__(self, canvas, boxes):
        self.boxes = boxes
        self.popup = Toplevel(canvas, borderwidth=3, relief="ridge")
        self.popup.title('Modify Categories')
        self.popup.grab_set()
        self.popup.resizable(False, False)
        self.fr = Frame(self.popup)
        self.fr.pack()
        self.my_text = Text(self.popup, width=20, height=1)
        self.my_text.pack(pady=5)
        self.lbx = Listbox(self.fr, width=20,
                           selectmode=EXTENDED, font=("Verdana", 16))
        self.lbx.bind('<<ListboxSelect>>', self.onselect)
        self.categories = boxes.categories

        self.my_text.bind('<Return>', lambda x=None: self.addCategory())
        self.my_text.config(fg='grey')
        self.my_text.insert(1.0, "Type category... ")
        self.my_text.bind("<FocusIn>", lambda x=None: self.handle_focus_in())
        self.my_text.bind("<FocusOut>", lambda x=None: self.handle_focus_out())

        self.lbx.pack(side=LEFT, fill="both", expand=True)
        for i in range(len(self.categories)):
            self.lbx.insert(i, self.categories[i])
        sbr = Scrollbar(self.fr)
        sbr.pack(side=RIGHT, fill="y")
        sbr.config(command=self.lbx.yview)
        self.lbx.config(yscrollcommand=sbr.set)

        frameButton = Frame(self.popup)
        self.buttonDelete = Button(
            frameButton,
            text="Delete",
            font=("Verdana", 16),
            command=lambda: self.deleteCategory()
        )
        self.buttonDelete.pack(side=LEFT)
        self.buttonDelete["state"] = "disabled"

        self.buttonAdd = Button(
            frameButton,
            text="Add",
            font=("Verdana", 16),
            command=lambda: self.addCategory()
        )
        self.buttonAdd.pack(side=LEFT)

        self.buttonModify = Button(
            frameButton,
            text="Modify",
            font=("Verdana", 16),
            command=lambda: self.modifyCategory()
        )
        self.buttonModify.pack(side=LEFT)
        self.buttonModify["state"] = "disabled"
        frameButton.pack()

    def modifyCategory(self):
        index = self.lbx.curselection()
        text = self.my_text.get(1.0, END).strip()
        if text != "" and text != "Type category..." and index != ():
            self.lbx.delete(index)
            self.lbx.insert(index, text)
            self.lbx.select_set(index)
            self.boxes.notifyCategoryModification(
                self.boxes.categories[index[0]], text)
            self.boxes.categories[index[0]] = text
            self.my_text.delete(1.0, END)

    def handle_focus_in(self):
        text = self.my_text.get(1.0, END).strip()
        if text == "" or text == "Type category...":
            self.my_text.delete(1.0, END)
            self.my_text.config(fg='black')

    def handle_focus_out(self):
        text = self.my_text.get(1.0, END).strip()
        if text == "":
            self.my_text.delete(1.0, END)
            self.my_text.config(fg='grey')
            self.my_text.insert(1.0, "Type category... ")

    def addCategory(self):
        text = self.my_text.get(1.0, END).strip()
        if text != "" and text != "Type category..." and not(text in self.boxes.categories):
            self.lbx.insert(END, text)
            self.boxes.categories.append(text)
            self.my_text.delete(1.0, END)

    def deleteCategory(self):
        index = self.lbx.curselection()
        if index != () and index[0] != 0:
            self.boxes.notifyCategoryDeletion(self.boxes.categories[index[0]])
            self.boxes.categories.pop(index[0])
            self.lbx.delete(index)
            self.buttonDelete["state"] = "disabled"
            self.buttonModify["state"] = "disabled"

    def onselect(self, event):
        index = event.widget.curselection()
        if index != () and index[0] != 0:
            self.buttonModify["state"] = "normal"
            self.buttonDelete["state"] = "normal"
        else:
            self.buttonModify["state"] = "disabled"
            self.buttonDelete["state"] = "disabled"
