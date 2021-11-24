from tkinter import *
import tkinter  
from PIL import ImageTk, Image  

root = Tk()
root.title("ImageAnnotator")
img = ImageTk.PhotoImage(Image.open("dessin.jpg"))  
canvas = Canvas(root, width = img.width(), height = img.height())  
canvas.pack()  
canvas.create_image(0, 0, anchor=NW, image=img)

class Box:
    def __init__(self):
        self.rect, self.x1, self.y1, self.x2, self.y2, self.categorie = None, None, None, None, None, None

    def click(self,event):
        self.x1 = event.x
        self.y1 = event.y

    def update(self,event,outline='red'):
        self.x2 = event.x
        self.y2 = event.y
        self.clear()
        self.draw(outline)

    def clear(self):
        if(self.rect is not None):
            canvas.delete(self.rect)
    
    def draw(self,outline):
        if(self.x1 is not None and self.y1 is not None and self.x2 is not None and self.y2 is not None):
            self.rect = canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline=outline, width=3)

class Boxes:
    def __init__(self):
        self.boxes = []
        self.currentBox = None

    def click(self,event):
        self.currentBox = Box()
        self.currentBox.click(event)

    def release(self,event):
        self.currentBox.update(event,'black')
        self.boxes.append(self.currentBox)
        self.popUp()

    def move(self,event):
        if(self.currentBox is not None):
            self.currentBox.update(event)

    def popUp(self):
        self.popup = PopUp()
        self.popup.setText(self.currentBox)
        self.popup.setOptionBar("one",["one", "two", "three"])
        self.popup.setButtons(self)

    def set(self):
        self.popup.clear()
        self.currentBox.clear()
        self.currentBox.draw("black")
        self.currentBox.categorie = self.popup.categorie.get()

    def delete(self):
        self.popup.clear()
        self.currentBox.clear()
        self.currentBox = None
        self.boxes.pop()

    def change(self,event):
        for boxe in self.boxes:
            if boxe.x1 <= event.x < boxe.x2 and boxe.y1 < event.y < boxe.y2:
                self.currentBox = boxe
                self.currentBox.clear()
                self.currentBox.draw("red")

                self.popup = PopUp()
                self.popup.setText(self.currentBox)
                self.popup.setOptionBar(self.currentBox.categorie,["one", "two", "three"])
                self.popup.setButtons(self)
                break

    def draw(self):
        for box in self.boxes:
            box.draw()

class PopUp:
    def __init__(self):
        self.canvas = Toplevel(root, borderwidth=3, relief="ridge")
        self.canvas.geometry("100x170")
        self.canvas.wm_attributes('-type', 'splash','-topmost', True)
        self.canvas.grid_rowconfigure(0,weight=1)
        self.canvas.grid_columnconfigure(0,weight=1)
        self.canvas.wait_visibility()
        self.canvas.grab_set()

    def setText(self,boxe):
        Label(self.canvas, text = f"Dimensions:\nx1: {boxe.x1}\ny1: {boxe.y1}\nx2: {boxe.x2}\ny2: {boxe.y2}", 
        font=('Mistral 10 bold'), justify=LEFT).grid(row=0, column=0 , columnspan=2)

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
            
boxes = Boxes()

root.bind('<Button-1>', boxes.click)
root.bind('<ButtonRelease-1>',boxes.release)
root.bind('<B1-Motion>', boxes.move)
root.bind('<Button-3>',boxes.change)

root.mainloop() 