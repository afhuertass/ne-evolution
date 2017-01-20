

import numpy as np
import pyscreenshot as ImGrab

class Capture():
    def __init__(self , x1 , y1 , x2 , y2):
        
        # 
        self.x1 = int(x1)
        self.x2 = int(x2)

        self.y1 = int(y1)
        self.y2 = int(y2)
        
        
        self.frame = np.zeros( (1,1) ) #dummy frame 

    def captureScreen(self):

        #self.frame = ImGrab.grab(bbox=(self.Point1[0], self.Point2[1] , self.Point2[0], self.Point2[1]) )
        self.frame = ImGrab.grab(bbox=(self.x1 , self.y1 , self.x2 , self.y2 ) )
        return self.frame
        
        
    
