class CameraType:
    
    word = ""
    color = ""
    servoAngle = -300
    
    def getType(self):
        return self.word
    
    def setType(self, word1):
        self.word = word1
    
    def setColor(self, color1):
        self.color = color1
    
    def getColor(self):
        return self.color

    def getAngle(self):
        return self.servoAngle
    
    def setAngle(self, angle1):
        self.servoAngle = angle1