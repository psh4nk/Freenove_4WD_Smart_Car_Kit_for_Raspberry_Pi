class CameraType:
    
    word = ""
    color = ""
    servoAngle = -300
    ballFound = False
    
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

    def getFound(self):
        return self.ballFound

    def setFound(self, found1):
        self.ballFound = found1