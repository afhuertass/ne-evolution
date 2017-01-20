import numpy as np

import cv2 
import matplotlib.pyplot as plt 

"""

Module for processing the frames captured by the screen and reducedem to a simplyfied version of the image, with the goal to use the proccesed image to feed the algorithm 

en principio es: 
Imagen sin procesar -> retornar imagen procesada


"""

class ImageProcessing():

    def __init__(self):

      

        self.min_hessian = 400 # parameters of the surf algorithm

        self.SURF = cv2.ORB_create(400)

        ## load trainImages images
        self.train_image1 = cv2.imread("./train_images/box.png" , 0 )
        
        #store shape for convinience 
        # key features and descriptors of the train images

        self.key_train , self.desc_train = self.SURF.detectAndCompute( self.train_image1 , None)

        print( self.desc_train.shape )
        # setup FLANN
        flann_index_kdtree = 1
        flann_index_lsh = 6
        #index_params = dict( algorithm = flann_index_kdtree, trees = 5 )
        index_params= dict(algorithm = flann_index_lsh ,
                   table_number =6, # 12
                   key_size = 12,     # 20
                           multi_probe_level = 1) #2
        
        search_params = dict(  checks = 50)
        
        self.flann = cv2.FlannBasedMatcher(index_params , {} )

        
        # las homography matrix will be keeped
        
        self.last_hmatrix = np.zeros( (3,3) )
        self.hq , self.wq = self.train_image1.shape[:2]


        self.templates_imgs = []
        
        
    def process(self , query_image=3):
        #query image is a image to be tested
        box = cv2.imread('./train_images/train_pigs.png' , 0 )
        
        query_image = cv2.imread('./train_images/query_1.jpg' , 0 )
        h1 , w1 = query_image.shape[:2]
        h2 , w2 = box.shape[:2]
        print(query_image.shape)
        print(box.shape)
        train_key , train_desc = self.SURF.detectAndCompute( box , None)
        query_key , query_desc = self.SURF.detectAndCompute( query_image , None)
        
        #
        response = np.zeros( (h1, w1 ), np.uint8 )
        response[:h1  ,  :w1 ] = query_image
        response = cv2.cvtColor(response , cv2.COLOR_GRAY2BGR )
        print("asdasda")
        print(response.shape)
        
        #matches = self.match_features( query_desc )

        #matches = self.flann.knnMatch( query_desc , trainDescriptors= train_desc , k = 2)
        matches = self.match_features( train_desc , query_desc )

        p1, p2 , kp_pairs  = self.filter_matches(  query_key , train_key , matches)
        print( len(p1) )
        if  len(p1) >= 4:
            H, status = cv2.findHomography( p1 , p2 ,  cv2.RANSAC , 5.0 )
            
        else:
            H , status = None , None

        if H is not None:
            # si hay homografia
            print("homeografia encontrada")
            
            print(w2)
            corners = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape( 1 , -1 , 2)
            
            corners = np.int32( cv2.perspectiveTransform(corners , H  ) )
            #corners = np.int32( cv2.perspectiveTransform(corners[:,:] , H  ) )
            cv2.polylines( response, corners, True, (255, 255, 255))
            
            cv2.fillPoly( response , [corners] , (255,100,255))


        img2 = cv2.drawKeypoints( box , train_key,0 ,color=(0,255,0), flags=0)
        
        plt.imshow( img2 )
        plt.show()
      
        
        return True

        
    def match_features(self ,  desc_train  ,desc_query ):
        
        matches = self.flann.knnMatch(  desc_train ,  trainDescriptors = desc_query , k = 2)
       
        #good_matches = filter(lambda x : x[0].distance<0.7*x[1].distance , matches)

        return matches
    
    def filter_matches(self, kp2, kp1, matches, ratio = 0.75):
        #kp2 = keypoints imagen QUERY
        #KP1 = keypoints imagen TRAINING
        mkp1, mkp2 = [], []
        print("len matches:" + str(len(matches)) )
        for m in matches:
            if len(m) == 2  and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
                
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        
        return p1, p2, list(kp_pairs)

    def process2(self):
        #second attempt for matching using template matching

        img = cv2.imread('./train_images/2048.png')
        
        img2 = img[:,:,2]
        img2 = img2 - cv2.erode( img2 , None)

        template = cv2.imread('./train_images/2.png')

        template = template[:, : , 2 ]
        
        template = template - cv2.erode( template, None )

        ccnorm = cv2.matchTemplate(img2 , template, cv2.TM_CCOEFF_NORMED )

        print(ccnorm )

        loc = np.where( ccnorm == ccnorm.max() )

        threshold = 0.4

        th , tw = template.shape[:2]

        for pt in zip(*loc[::-1]):
            if ccnorm[pt[::-1]] < threshold:
                continue
            print("aasdasdsa")
            cv2.rectangle(img, pt, (pt[0] + tw, pt[1] + th),
                        (0, 0, 255), 2)

            
        plt.imshow(img)
        plt.show()


    def process3(self, img):
            # alto y ancho de la casilla
        img_opencv = np.array(img)
        
        self.load_templates()
        data = [] 

        #img = cv2.imread( './train_images/test.png')
        img = img_opencv.copy()
        img = img[: , : , 2] # canal rojo
        
        img = img - cv2.erode(img,None) # gradiente morfologico
        
        h , w = 125 , 125 
        px , py =  0 , 0 
        for col in range(0,4):
            
            for row in range(0,4):
                
                cell = img[ px : px + w , py : py  + h     ]
                # cell
                found_number = False
                for template in self.templates_imgs:
                    is_in = self.is_in_template( cell , template[0] )
                    if is_in:
                        
                        #print(template[1] )
                        data.append( template[1] )
                        found_number = True
                        break
                    
                if not found_number :
                    data.append( 0 )
                    
                py  = py + h
                """
                print(px)
                print(py)
                plt.imshow(cell)
                plt.show()
                """
            px = px  + w
            py = 0
        return data
            # now the list datas contain the data for the learning algorithm
            
    def  is_in_template(self , cell , template):
        
        ccnorm = cv2.matchTemplate(cell , template, cv2.TM_CCOEFF_NORMED )
        #print(ccnorm )
        loc = np.where( ccnorm == ccnorm.max() )

        threshold = 0.4

        th , tw = template.shape[:2]

        for pt in zip(*loc[::-1]):
            if ccnorm[pt[::-1]] < threshold:
                continue
            return True
           
                        
        return False
    def load_templates(self):

        templates_names = [ [ "./train_images/templates/2.png", 2 ] ,
                            [ "./train_images/templates/4.png", 4 ] ,
                            [ "./train_images/templates/8.png", 8 ] ,
                            [ "./train_images/templates/16.png", 16 ] ,
                            [ "./train_images/templates/32.png", 32 ] ,
                            [ "./train_images/templates/64.png", 64 ] ,
                            [ "./train_images/templates/128.png", 128 ] ,
                            [ "./train_images/templates/256.png", 256 ] ,
                            [ "./train_images/templates/512.png", 512 ]
                                    ]
         #"./train_images/templates/4.png" ,
         #"./train_images/templates/8.png",
         #"./train_images/templates/16.png"
        for temp_name in templates_names:

            im_template = cv2.imread( temp_name[0] )
            im_template = im_template[: , : , 2] # canal rojo
            # gradiente morfologico 
            im_template = im_template - cv2.erode( im_template ,None )
            # agregar la imagen a la lista
            self.templates_imgs.append( [im_template , temp_name[1]  ] )
            
        
    def draw(self, im1 , im2 , kp_pairs ):

        print ("fuckkkkkkk")
