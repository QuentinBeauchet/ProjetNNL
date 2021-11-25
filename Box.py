from shapely.geometry import box

class Box:
    def __init__(self,canvas):
        self.rect, self.polygon, self.clickEvent, self.categorie = None, None, None, None
        self.canvas = canvas

    def __lt__(self, other):
        return self.polygon.area < other.polygon.area

    def click(self,event):
        self.clickEvent = event

    def update(self,event,outline='cyan'):
        self.polygon = box(self.clickEvent.x,self.clickEvent.y,event.x,event.y)
        self.draw(outline)

    def clear(self):
        if(self.rect is not None):
            self.canvas.delete(self.rect)

    def coords(self):
        x,y = self.polygon.exterior.coords.xy
        return (x[1],y[1],x[3],y[3])

    def size(self):
        x1,y1,x2,y2 = self.coords()
        return (abs(x2-x1),abs(y2-y1))

    def isValid(self):
        width, height = self.size()
        return self.polygon.area>=40 and width>=5 and height>=5

    def draw(self,outline):
        if(self.polygon is not None):
            self.clear()
            self.rect = self.canvas.create_rectangle(self.coords(), outline=(outline if self.isValid() else "red"), width=2)