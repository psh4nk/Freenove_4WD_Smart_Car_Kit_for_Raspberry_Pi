class imageGetter:
    image = None
    cx = None
    cy = None
    pixel_middle = []

    def getImage(self):
        return self.image

    def setImage(self, image1):
        self.image = image1
    
    def getPixelMiddle(self):
        return self.pixel_middle

    def setPixelMiddle(self, pixelMiddle):
        self.pixel_middle = pixelMiddle

    def getCX(self):
        return self.cx
    
    def getCY(self):
        return self.cy

    def setCX(self, cx1):
        self.cx = cx1

    def setCY(self, cy1):
        self.cy = cy1
    