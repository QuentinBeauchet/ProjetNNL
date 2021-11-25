from tkinter import Label
from tkinter import Toplevel
from tkinter import StringVar
from tkinter import OptionMenu
from tkinter import Button
from tkinter import LEFT
import platform

class PopUp:
    def __init__(self,canvas):
        self.canvas = Toplevel(canvas, borderwidth=3, relief="ridge")
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