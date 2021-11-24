from tkinter import *  
from PIL import ImageTk, Image  

root = Tk()  
img = ImageTk.PhotoImage(Image.open("dessin.jpg"))  
canvas = Canvas(root, width = img.width(), height = img.height())  
canvas.pack()  
canvas.create_image(0, 0, anchor=NW, image=img)

class Box:
    def __init__(self):
        self.rect, self.x1, self.y1, self.x2, self.y2 = None, None, None, None, None

    def click(self,event):
        self.x1 = event.x
        self.y1 = event.y

    def update(self,canvas,event,outline='red'):
        if(self.rect is not None):
            canvas.delete(self.rect)
        self.x2 = event.x
        self.y2 = event.y
        self.draw(outline)
    
    def draw(self,outline):
        if(self.x1 is not None and self.y1 is not None and self.x2 is not None and self.y2 is not None):
            self.rect = canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline=outline, width=3)

class Boxes:
    def __init__(self,canvas):
        self.boxes = []
        self.canvas = canvas
        self.currentBox = None

    def click(self,event):
        self.currentBox = Box()
        self.currentBox.click(event)

    def release(self,event):
        self.currentBox.update(canvas,event,'black')
        self.boxes.append(self.currentBox)
        self.popUp()

    def move(self,event):
        if(self.currentBox is not None):
            self.currentBox.update(canvas,event)

    def popUp(self):
        global root
        top = Toplevel(root)
        top.geometry("200x100")
        top.title("Parametres de la selection")
        text = f"Selection:\nx1: {self.currentBox.x1}\ny1: {self.currentBox.y1}\nx2: {self.currentBox.x2}\ny2: {self.currentBox.y2}\n"
        Label(top, text = text, font=('Mistral 10 bold')).pack(anchor="center")

    def draw(self):
        for box in self.boxes:
            box.draw()
            
boxes = Boxes(canvas)

root.bind('<Button-1>', boxes.click)
root.bind('<ButtonRelease-1>',boxes.release)
root.bind('<B1-Motion>', boxes.move)

root.mainloop() 