
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By




import image_processing  as ImProc
import capture as capture


import numpy as np

class DriverData():

    def __init__(self):

        ## this class is the driver of data.
        """
        It get asked for the data from the we, it gets the data
        aka captures de image and process it, and returns the list

        if will also had a method for asking the score.

        and an method to send keystrokes 

        """
        # object for processing the images
        self.image_processer = ImProc.ImageProcessing()
        # we need the points to create the capturer
        x1 , y1 = 580 , 360
        x2 , y2 = 1080 , 860
        self.capturer = capture.Capture(x1,y1,x2 , y2 )

        self.url =  "https://gabrielecirulli.github.io/2048/"
        # the web driver
        self.driver = webdriver.Chrome('/home/andresh/Descargas/chrome_driver/chromedriver' )

        self.driver.get(self.url)

        self.driver.maximize_window()
        time.sleep(3)
        #close the ad button
        close = self.driver.find_element_by_class_name("notice-close-button")
        close.click()
        
        

        # body element to send key strokes
        self.body = self.driver.find_elements(By.XPATH , '//body')[0]
        
        # restart
        self.restart = self.driver.find_element_by_class_name("restart-button")
        
    def get_train_list(self):

        # it should return a list from the image
        self.driver.execute_script( 'window.focus();' )
        img = self.capturer.captureScreen()
        
        data_train = self.image_processer.process3(img) 
        # and the we
        
        return data_train

    def read_score(self):

        
        score_element = self.driver.find_element_by_class_name("score-container")
        # get the score as an int
        return ( int( score_element.text ) )
    

    def new_game(self):

        self.restart.click()
        time.sleep(0.2)

    def handle_response(self , int_key):

        self.send_keystroke(self, int_key)

        return True 
    def send_keystroke(self, key):

        if key == 0:
            self.body.send_keys( Keys.ARROW_LEFT )
            
        if key == 1:
            self.body.send_keys( Keys.ARROW_RIGHT )
           
        if key == 2:
            self.body.send_keys( Keys.ARROW_UP )
            
        if key == 3:
            self.body.send_keys( Keys.ARROW_DOWN )
            
        
        time.sleep(0.1)
        return False

    def close_driver(self):
        
        self.driver.quit()


class XORDriver():

    def __init__(self):

        # compuerta XOR
        #
        self.inputs = []
        self.outputs = np.array( [ 0, 1 , 1 , 0] , np.float64 )
        l1 = [ 1.0 , 1.0]
        l2 = [ 1.0 , 0.0 ]
        l3 = [ 0.0 , 1.0]
        l4 = [ 1.0 , 1.0]
        
        self.inputs.append(l1)
        self.inputs.append(l2)
        self.inputs.append(l3)
        self.inputs.append(l4)
        
        
    def get_train_list(self, index_op = 0):

        
        return self.inputs[ index_op ]
    
    # TODO: CHANGE NAME OF THIS
    def read_score(self, outputs):

        
        fit = np.power( self.outputs - np.array( outputs  , np.float64) , 2 )
        print("fiiitt")
        print(outputs)
        print(self.outputs)
        print(  fit ) 
        fit = 10 - np.fabs(np.sqrt(  np.sum( fit ) ) )
        fit = np.power( fit , 2 )
       
        return float(fit)
    
    def handle_response( self , none):

        return True

    def close_driver(self):

        #nada
        return True
    def new_game(self):

        return True
