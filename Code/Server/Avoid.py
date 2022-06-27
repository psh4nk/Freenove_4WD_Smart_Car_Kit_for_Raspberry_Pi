import time
from Motor import *
import RPi.GPIO as GPIO
class Avoid:
    def __init__(self):
        self.IR01 = 14
        self.IR02 = 15
        self.IR03 = 23
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.IR01,GPIO.IN)
        GPIO.setup(self.IR02,GPIO.IN)
        GPIO.setup(self.IR03,GPIO.IN)
    def run(self):
        while True:
            self.LMR=0x00
            if GPIO.input(self.IR01)==True:
                self.LMR=(self.LMR | 4)
            if GPIO.input(self.IR02)==True:
                self.LMR=(self.LMR | 2)
            if GPIO.input(self.IR03)==True:
                self.LMR=(self.LMR | 1)
            if self.LMR==0:
                #pass
                PWM.setMotorModel(800,800,800,800)
            elif self.LMR > 0:
                PWM.setMotorModel(-600,-600,-600,-600)
                time.sleep(2)
                PWM.setMotorModel(2500, 2500, -1500, -1500)
                time.sleep(1)
            
infrared=Avoid()
# Main program logic follows:
if __name__ == '__main__':
    print ('Program is starting ... ')
    try:
        Avoid.run()
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child program  will be  executed.
        PWM.setMotorModel(0,0,0,0)
